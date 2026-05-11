 # Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE
import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from functools import partial
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from lit_gpt.model import GPT, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils import chunked_cross_entropy, num_parameters
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from lit_gpt import FusedCrossEntropyLoss
import random
import os
import argparse
from data import get_stateful_stream_tok_dataset
import time
import torch.multiprocessing as mp
import shutil
from distutils.dir_util import copy_tree

_TRAIN_START_TIME = time.time()

os.environ["TRITON_CACHE_MANAGER"] = "cache:ParallelFileCacheManager"


def main(args):
    if args.debug:
        wandb_logger = WandbLogger(project="llm_next_gen", mode='disabled', name=args.exp_name, id=args.exp_name, save_dir=args.wandb_dir, dir=args.wandb_dir, version=args.exp_name, group="debug")
    else:
        wandb_logger = WandbLogger(project="llm_next_gen", name=args.exp_name, id=args.exp_name, save_dir=args.wandb_dir, dir=args.wandb_dir, version=args.exp_name, group=args.exp_group)

    if args.interactive_job:
        strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full")
    else:
        strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy='HYBRID_SHARD')
    fabric = L.Fabric(devices=devices, strategy=strategy, precision="bf16-mixed", loggers=[wandb_logger])
    fabric.launch()
    # fix seed in the very beginning
    fabric.seed_everything(args.seed)  # same seed for every process to init model (FSDP)
    fabric.print("##### Infra Details #####")
    fabric.print(f"Number of Nodes: {args.nodes}")
    fabric.print(f"Number of GPUs: {fabric.world_size}")
    fabric.print("##### Training Details #####")
    fabric.print(f"Maximum number of training tokens: {args.max_tokens}")
    fabric.print(f"Maximum training time: {args.actual_train_time/60.0} min")
    fabric.print(f"Micro batch size: {args.micro_batch_size}")
    fabric.print(f"Batch size: {args.batch_size}")
         
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME], device=fabric.device, dtype=torch.int64)
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()    
    if fabric.global_rank == 0:
        fabric.print(args)
    fabric.logger.log_hyperparams(args)
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=args.log_iter_interval)

    if os.path.exists(args.out_dir):
        args.resume = True
        print('Resuming from {}'.format(args.out_dir))
    else:
        if fabric.global_rank == 0:
            os.makedirs(args.out_dir)
            target_litgpt_save_dir = os.path.join(args.out_dir, 'lit_gpt')
            target_bash_scripts_save_dir = os.path.join(args.out_dir, 'bash_scripts')
            target_pretrain_file = os.path.join(args.out_dir, 'pretrain.py')
            os.makedirs(target_litgpt_save_dir)
            os.makedirs(target_bash_scripts_save_dir)
            if not args.debug:
                copy_tree(os.path.join(wd, 'modern_litgpt/lit_gpt'), target_litgpt_save_dir)
                copy_tree(os.path.join(wd, "modern_litgpt/bash_scripts"), target_bash_scripts_save_dir)
                shutil.copyfile(os.path.join(wd, "modern_litgpt/pretrain.py"), target_pretrain_file)                
    
    config = Config.from_name(args.model_name)
    if args.use_stream_tok:
        tokenizer = AutoTokenizer.from_pretrained('/lustre/fsw/portfolios/nvr/users/ahatamizadeh/models/tokenizers') 
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = 999999999
        train_dataloader = get_stateful_stream_tok_dataset(corpus_name=args.corpus_name, path=args.train_data_dir_raw, split='train', tokenizer=tokenizer, block_size=config.block_size+1, rank=fabric.global_rank, world_size=fabric.world_size, batch_size=args.micro_batch_size, num_workers=args.train_num_workers) 
        val_dataloader = get_stateful_stream_tok_dataset(corpus_name=args.corpus_name, path=args.val_data_dir_raw, split=args.val_type, tokenizer=tokenizer, block_size=16384+1, rank=fabric.global_rank, world_size=fabric.world_size, batch_size=args.micro_batch_size // 2, num_workers=args.val_num_workers)

    else:
        train_dataloader, val_dataloader = create_dataloaders(
        batch_size=args.micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        seed=args.seed,
        )
        if val_dataloader is None:
            train_dataloader = fabric.setup_dataloaders(train_dataloader)
        else:
            train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    if args.val_type != 'val_sampled':
        val_dataloader = None
        
    if fabric.global_rank == 0:
        fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))
    
    if fabric.global_rank == 0:
        fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
        # we ignore the embedding & lm head parameter, which is standard 
        fabric.print(f"Total parameters {num_parameters(model.transformer.h):,}")
        fabric.print(model)
    
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), fused=True
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": args.hparams, "iter_num": 0, "step_count": 0}

    if args.resume:
        try:
            resume = os.path.join(args.out_dir, "latest-model-ckpt.pth")
            if fabric.global_rank == 0:
                fabric.print(f"Resuming training from {resume}")
            fabric.load(resume, state)
            fabric.print(f"Successfully resumed from {resume}")
        except:
            fabric.print(f"Failed to resume from {resume}")
            args.resume = False
    train_time = time.perf_counter()
    train(args, _TRAIN_START_TIME, fabric, state, train_dataloader, val_dataloader, monitor, args.resume)
    if fabric.global_rank == 0:
        fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        if fabric.global_rank == 0:
            fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(args, _TRAIN_START_TIME, fabric, state, train_dataloader, val_dataloader, monitor, resume):
    
    model = state["model"]
    optimizer = state["optimizer"]

    total_lengths = 0
    total_t0 = time.perf_counter()    
    max_tokens_per_device = args.max_tokens // fabric.world_size
    tokens_per_iter = args.micro_batch_size * model.config.block_size
    max_iters = max_tokens_per_device // tokens_per_iter
    warmup_iters = args.warmup_tokens // fabric.world_size // tokens_per_iter
    initial_iter = state["iter_num"]
    curr_iter = 0
    loss_func = FusedCrossEntropyLoss()    

    if resume:
        if args.use_stream_tok:
            try:
                if fabric.world_size <= 1:
                    data_state_path = os.path.join(args.out_dir,"latest-data-state-ckpt.pth")
                else:
                    data_state_path = os.path.join(args.out_dir,f"latest-data-states-rank-{fabric.global_rank}-ckpt.pth")
                train_dataloader.load_state_dict(torch.load(data_state_path))                                                
                if fabric.global_rank == 0:
                    fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
                resume = False
            except:
                fabric.print(f"Failed to resume dataloader from {args.out_dir}")
                raise KeyError("Failed to resume dataloader.. Please retrain from scratch.")

    tokens = 0
    train_t0 = time.perf_counter()
    
    if args.eval_before_training:
        fabric.print("Do validation before training:")
        val_loss = validate(args, fabric, model, val_dataloader, None)
        for i in range(args.num_extrapol):
            if fabric.global_rank == 0:
                fabric.print(f"step {state['iter_num']} {i+1} x: val loss {val_loss[i]:.4f}")
    
    def save_checkpoint(final=False):
        name = 'latest' if not final else 'final'
        checkpoint_path = os.path.join(args.out_dir,f"{name}-model-ckpt.pth")
        fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
        if not final:
            fabric.save(checkpoint_path, state)
        else:
            # we are not interested in the optimizer state for the final checkpoint 
            state['optimizer'] = None
            fabric.save(checkpoint_path, state)

        if args.use_stream_tok and not final:
            if fabric.world_size <= 1:
                checkpoint_path = os.path.join(args.out_dir,f"latest-data-state-ckpt.pth")
            else:
                checkpoint_path = os.path.join(args.out_dir,f"latest-data-states-rank-{fabric.global_rank}-ckpt.pth")
            torch.save(train_dataloader.state_dict(), checkpoint_path)                
            fabric.print(f"Dataloader state checkpoint saved")

    for train_data in train_dataloader:
        # per gpu
        tokens += model.config.block_size * args.micro_batch_size
        if resume and not args.use_stream_tok:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                if fabric.global_rank == 0:
                    fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))

        if state["iter_num"] >= max_iters:
            break
    
        iter_t0 = time.perf_counter()
        if args.use_stream_tok:
            input_ids = train_data['input_ids'][:, 0 : model.config.block_size].contiguous().to(fabric.device)
            targets = train_data['labels'][:, 1 : model.config.block_size + 1].contiguous().to(fabric.device)
        else:
            input_ids = train_data[:, 0 : model.config.block_size].contiguous()
            targets = train_data[:, 1 : model.config.block_size + 1].contiguous()

        lr = get_lr(args, state["iter_num"], warmup_iters, max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        is_accumulating = (state["iter_num"] + 1) % args.gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = loss_func(logits, targets)
            # Check if loss is NaN
            if torch.isnan(loss):
                # Create a debug directory if it doesn't exist
                debug_dir = "./logs/debug"
                os.makedirs(debug_dir, exist_ok=True)
                
                # Save the relevant tensors
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                torch.save({
                    'input_ids': input_ids,
                    'logits': logits,
                    'targets': targets,
                    'loss': loss
                }, os.path.join(debug_dir, f'nan_tensors_{timestamp}.pt'))
                
                print(f"NaN loss detected! Tensors saved to {debug_dir}/nan_tensors_{timestamp}.pt")
            fabric.backward(loss / args.gradient_accumulation_steps)


        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        state["iter_num"] += 1
        # input_id: B L 
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        if fabric.global_rank == 0 and state["iter_num"] % 10 == 0:
            total_tokens = model.config.block_size * state["iter_num"] * args.micro_batch_size * fabric.world_size / 1e9
            fabric.print(
                    f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                    f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                    f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                    f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
                    f" total training throughput {tokens / (t1 - train_t0) / 1e3:.2f}K tokens/s per GPU."
                    f" total trained tokens: {total_tokens} B tokens"
                    f" peak memory allocate {torch.cuda.memory_stats(0)['allocated_bytes.all.peak'] / 1e9} GB"
                )           
            
        estimated_flops = 1
        monitor.on_train_batch_end(
            state["iter_num"] * args.micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item()
        )        

        # Exiting based on duration. 
        # credits: https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/e52bdabbde3c6895aceb76c1bced295c2646121f/megatron/training.py#L985-L998
        if not is_accumulating and args.actual_train_time:
            train_time = (time.time() - _TRAIN_START_TIME)
            # start monitoring sync
            done_cuda = torch.tensor([train_time > args.actual_train_time], device=fabric.device, dtype=torch.int)
            # force synchronization.
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                fabric.print(f"Training time {train_time/60.0} min, Reach time limit. Exiting ...")
                save_checkpoint()
                sys.exit()

        if not is_accumulating and state["step_count"] % args.save_step_interval == 0:
            save_checkpoint()

        # First save ckpt then do eval in case ckpt is not saved in time
        if val_dataloader is not None and not is_accumulating and state["step_count"] % args.eval_step_interval == 0:            
            t0 = time.perf_counter()
            val_loss = validate(args, fabric, model, val_dataloader, args.eval_iters)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            for i in range(args.num_extrapol):
                if fabric.global_rank == 0:
                    fabric.print(f"step {state['iter_num']} {i+1} x: val loss {val_loss[i]:.4f}, val time: {t1 * 1000:.2f}ms")        
                    fabric.log_dict({"metric/val_loss@"+str(i+1)+"x": val_loss[i].item()}, state["step_count"])
                    fabric.log_dict({"metric/val_ppl@"+str(i+1)+"x": math.exp(val_loss[i].item())}, state["step_count"])

            fabric.barrier()
    
    save_checkpoint(final=True)

# each gpu will run validation on the entire val_dataset to avoid headache.
@torch.no_grad()
def validate(args, fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, eval_iters=100) -> torch.Tensor:
    if fabric.global_rank == 0:
        fabric.print("Validating ...")
    model.eval()
    #eval_iters = eval_iters if eval_iters is not None else args.eval_iters
    losses = torch.zeros(args.num_extrapol, device=fabric.device, dtype=torch.float64)
    num_sample = 0
    for k, val_data in enumerate(val_dataloader):
        num_sample += 1
        # if k >= eval_iters:
            # break        
        for i, length in enumerate([4096, 8192, 12288, 16384]):   #[2048, 4096, 8192, 16384]
            if args.use_stream_tok:
                input_ids = val_data['input_ids'][:, 0:length].contiguous().to(fabric.device)
                targets = val_data['labels'][:, 1:length + 1].contiguous().to(fabric.device)
            else:
                input_ids = val_data[:, 0 : length].contiguous()
                targets = val_data[:, 1 : length + 1].contiguous()
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            # running average to avoid overflow
            losses[i] += (loss.item() - losses[i]) / num_sample
    fabric.print(f"Validation loss: {losses}")
    model.train()
    return losses


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        #filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        filenames = sorted(glob.glob(os.path.join(data_dir,f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)
        if split != "train":
            n_chunks = - (8 // -nodes) # ceil division
        else:
            n_chunks = 8
        dataset = PackedDataset(
            filenames,
            n_chunks=n_chunks,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            batch_size=- (batch_size // -2), # ceil division
            block_size=  16384 + 1, #num_extrapol * block_size + 1, # val 4* extrapolation
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(args, it: int, warmup_iters: int, max_iters: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    devices = torch.cuda.device_count() or 1
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='LLM Training')
    group = parser.add_argument_group('hyperparameters')
    group.add_argument('--output_root', default='', type=str, help='output root directory')
    group.add_argument('--wandb_dir', default='', type=str, help='wandb directory')
    group.add_argument('--train_data_dir', default='', type=str, help='training data directory')
    group.add_argument('--corpus_name', default='slimpajama', type=str, help='corpus name')
    group.add_argument('--train_data_dir_raw', default='', type=str, help='training data directory (raw file for stream tok)')
    group.add_argument('--val_data_dir', default='', type=str, help='validation data directory')
    group.add_argument('--val_data_dir_raw', default='', type=str, help='validation data directory (raw file for stream tok)')
    group.add_argument('--model_name', default='Samba_421M', type=str, help='model name')
    group.add_argument('--exp_name', default='', type=str, help='experiment name')
    group.add_argument('--exp_group', default='', type=str, help='experiment group name')
    group.add_argument('--train_config', default='tsz512x4k_20B', type=str, help='training config')
    group.add_argument('--resume', action='store_true', default=False, help='resume flag')
    group.add_argument('--debug', action='store_true', default=False, help='debug flag')
    group.add_argument('--interactive_job', action='store_true', default=False, help='debug flag')
    group.add_argument('--use_stream_tok', action='store_true', default=False, help='resume flag')
    group.add_argument('--tokenizer_name', type=str, default='TinyLlama/TinyLlama_v1.1')
    group.add_argument('--tokenizer_path', type=str, default='/lustre/fsw/portfolios/nvr/projects/nvr_lpr_nvgptvision/datasets/llm_next_gen/tokenizers/TinyLlama/TinyLlama_v1.1')
    group.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate')
    group.add_argument('--total_evals', type=int, default=400, help='total number of evals')
    group.add_argument('--eval_iters', type=int, default=15, help='number of evaluation iterations')
    group.add_argument('--log_step_interval', type=int, default=10, help='log_step_interval')
    group.add_argument('--save_step_interval', type=int, default=1000, help='save_step_interval')
    group.add_argument('--eval_step_interval', type=int, default=1000, help='eval_step_interval')
    group.add_argument('--seed', type=int, default=3407, help='seed')
    group.add_argument('--num_extrapol', type=int, default=4, help='num_extrapol')
    group.add_argument('--weight_decay', type=float, default=1e-1, help='weight decay')
    group.add_argument('--beta1', type=float, default=0.9, help='beta1')
    group.add_argument('--beta2', type=float, default=0.95, help='beta2')
    group.add_argument('--grad_clip', type=float, default=1.0, help='gradient clip')
    group.add_argument('--val_type', default='val_sampled', type=str, help='choose between val_sampled and val for validation type')
    group.add_argument('--eval_before_training', action='store_true', default=False, help='do validation before the training starts')
    group.add_argument('--nnodes', type=int, default=None, help='number of nodes')
    group.add_argument('--train_num_workers', type=int, default=8)
    group.add_argument('--val_num_workers', type=int, default=1)
    group.add_argument('--actual_train_time', type=int, default=235, help='actual training time in mins')
    group.add_argument('--micro_batch_size', type=int, default=0, help='micro batch size')

    args = parser.parse_args()
    name = args.train_config +"_" + args.exp_name
    args.out_dir = args.output_root + '/outputs/' + name
    args.wandb_dir = args.output_root + '/wandb/' + name

    train_data_config = [("train_slim", 1.0)]
    val_data_config = [("validation", 1.0)]

    nodes = int(os.getenv("SLURM_NNODES"))
    args.nodes = nodes

    micro_batch_size = 8  

    if "20B" in name:
        max_tokens = int(1e11) // 5 # 20 billion
    elif "100B" in name:
        max_tokens = int(1e11) # 100 billion
    elif "50B" in name:
        max_tokens = int(1e11) // 2 # 50 billion
    elif "30B" in name:
        max_tokens = int(3e10) # 30 b
    elif "15B" in name:
        max_tokens = int(3e10) // 2 # 30 b
    else:
        raise ValueError("Unknown training token config")
    
    if "512x4k" in name:
        micro_batch_size = 8
        global_batch_size = 512 // nodes 
    elif "1024x4k" in name:
        micro_batch_size = 8 
        global_batch_size = 1024 // nodes

    elif "256x8k" in name:
        #8k
        global_batch_size = 256 // nodes
        micro_batch_size = 8 

    elif "128x16k" in name:
        #16k
        global_batch_size = 128 // nodes
        micro_batch_size = 4

    elif "64x32k" in name:
        #32k
        global_batch_size = 64 // nodes
        micro_batch_size = 2 
        
    elif "1024x2k" in name:
        #2k
        global_batch_size = 1024 // nodes
        micro_batch_size = 32
    
    if "1.3B" in name:
        micro_batch_size = micro_batch_size // 2 
    
    micro_batch_size = max(1, micro_batch_size)
    args.min_lr = args.learning_rate / 10
    args.batch_size = global_batch_size // devices

    gradient_accumulation_steps = args.batch_size // micro_batch_size

    assert gradient_accumulation_steps > 0
    log_iter_interval = args.log_step_interval * gradient_accumulation_steps
    args.gradient_accumulation_steps = gradient_accumulation_steps

    args.actual_train_time = args.actual_train_time * 60 # convert to seconds
    args.warmup_tokens = int(max_tokens * 0.01)
    args.max_tokens = max_tokens
    if args.micro_batch_size == 0:
        args.micro_batch_size = micro_batch_size
    args.log_iter_interval= log_iter_interval
    hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
    args.hparams = hparams
    main(args)
