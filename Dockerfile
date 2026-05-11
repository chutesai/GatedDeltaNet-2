FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

ENV TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX"
ENV MAX_JOBS=2
ENV NVCC_THREADS=2
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libz3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN echo "torch==2.9.0" > /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt

RUN pip install --no-cache-dir ninja packaging einops transformers

RUN pip install --no-cache-dir \
    "apache-tvm-ffi>=0.1.6,<0.1.10" \
    torch-c-dlpack-ext \
    cloudpickle \
    ml-dtypes \
    psutil \
    "z3-solver>=4.13.0,<4.15.5" \
    "nvidia-cutlass-dsl>=4.4.2"

RUN pip install --no-cache-dir tilelang==0.1.8 --no-deps
RUN pip install --no-cache-dir quack-kernels==0.3.4 --no-deps

RUN pip install --no-cache-dir "causal-conv1d==1.6.1" --no-build-isolation

RUN MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --no-deps \
    --no-build-isolation \
    git+https://github.com/state-spaces/mamba.git

RUN pip install --no-cache-dir "flash-attn==2.8.3" --no-build-isolation

RUN pip install --no-cache-dir -U \
    git+https://github.com/sustcsonglin/flash-linear-attention \
    --no-build-isolation

RUN pip install --no-cache-dir \
    lightning==2.1.2 "lightning[app]" "lightning[data]" \
    jsonargparse[signatures] tokenizers sentencepiece wandb torchmetrics \
    tensorboard zstandard pandas pyarrow huggingface_hub

RUN pip install --no-cache-dir datasets==2.20.0 xformers "tomli>=1.1.0"

RUN pip install --no-cache-dir \
    braceexpand smart_open opt_einsum cbor2 isort pytest mypy \
    mosaicml-streaming

RUN pip uninstall torchdata -y || true
RUN pip install --pre --no-cache-dir torchdata \
    --index-url https://download.pytorch.org/whl/nightly

RUN pip install --no-cache-dir lm-eval