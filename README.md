# AssemblyStyleTransfer

## Setup

Install
```
conda create -n AssemblyStyleTransfer python=3.10 pytorch=2.0 torchvision=0.15 torchaudio=2.0 torchtext=0.15 pytorch-cuda=11.8 transformers=4.28 datasets=2.12 tokenizers=0.11 accelerate=0.19 capstone=4.0 psutil=5.9 pefile=2022.5.30 scikit-learn=1.2 -c pytorch -c nvidia -c huggingface -c conda-forge -c anaconda
conda activate AssemblyStyleTransfer
pip install evaluate, r2pipe

```

## Steps

### Prepare

### Explain

### Prepare

### Preprocess

### Pretrain

### Train

## TODO

- Improve organization of preprocess script's main function.
- CLI functions should print start, stop, and job info to facilitate cluster usage.
- Improve logging and enable various log levels.


## For multi GPU computing
https://pytorch.org/docs/stable/elastic/run.html
torchrun \
--nproc-per-node 1 \
--rdzv-endpoint=localhost:29501 \
