# 23-3 Capstone Design Project - Optimus GPrime

<div style="display:flex; flex-direction:row;">
  <img src="https://img.shields.io/badge/C%2B%2B-%2300599C?style=flat-square&logo=cplusplus&logoColor=white"/> 
  <img src="https://img.shields.io/badge/CUDA-%23A8B9CC?style=flat-square&logo=nvidia&logoColor=white"/> 
  <img src="https://img.shields.io/badge/Python-%233776AB?style=flat-square&logo=python&logoColor=white"/>     
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/> 
  <img src="https://img.shields.io/badge/Bash-%234EAA25?style=flat-square&logo=gnubash&logoColor=white"/>
</div>

## Project Overview
Welcome to the Optimus GPrime Capstone Design Project! This GitHub repository contains the code and instructions for setting up and running experiments related to our capstone design project. The primary focus is on implementing and benchmarking vLLM, a language model, with specific configurations and datasets.

## Setup Instructions

### 1. Execute the following script files.
```
# Download conda.
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
./Anaconda3-2023.09-0-Linux-x86_64.sh
cd anaconda3/
source ~/.bashrc

# For personalization.
CONDA_ENV_NAME=""

conda create --name $CONDA_ENV_NAME python=3.8
conda activate $CONDA_ENV_NAME

# Install necessary libraries.
cd
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
conda install -c "nvidia/label/cuda-12.1.0" cudnn
conda install pytorch==2.1.0 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# Download vLLM.
git clone https://github.com/vllm-project/vllm.git

# Build and install vLLM from the downloaded sources.
cd vllm
pip install -e .
pip install -r requirements.txt
```

### 2. Add block size configurations of vLLM.
Modify the block size configurations in /vllm/csrc/attention/attention_kernels.cu as follows
```
#define CALL_V1_LAUNCHER_BLOCK_SIZE(T) \
  switch (block_size) { \
    case 1: \
      CALL_V1_LAUNCHER(T, 1); \
      break; \
    // ... (cases 2, 4, 8, 16, 32, 64, 128) ...
    default: \
      TORCH_CHECK(false, "EB: Unsupported block size: ", block_size); \
      break; \
   }
```

### 3. Build and install packages of vLLM.
```
cd ~/vllm
python3 setup.py build
python3 setup.py install
```

### 4. Download Datasets
```
cd ~/vllm/benchmarks
mkdir datasets
cd datasets

# Download ShareGPT dataset.
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
mv ShareGPT_V3_unfiltered_cleaned_split.json ShareGPT.json

# Download Alpaca dataset.
wget https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json
mv alpaca_data_cleaned.json alpaca.json
```

### 5. Create and execute benchmarks.
Note: Before executing the benchmark, ensure that you have configured the settings in the mentioned files according to your requirements.
```
# Before you execute this file, please check configuration settings.
# 1) swap and recomputation configuration -> vllm/vllm/core/scheduler.py
# 2) block size -> vllm/vllm/engine/arg_utils.py
# 3) swap space -> vllm/vllm/entrypoints/llm.py

# Set your test model and dataset.
MODEL="llama-13b"
DATASET="alpaca"

# Set path.
BENCH_FILE=$DATASET"_throughput.py" # Benchmark file name
MODEL_PATH="models/"$MODEL
DATASET_PATH="datasets/"$DATASET".json"

# Set Config.
BLOCKSIZE="1" # Default size is 16
PREEMPT_MODE="swap" # recompute or swap

export TOKENIZERS_PARALLELISM=TRUE # EB: Set as distributed mode
nohup python3 -u $BENCH_FILE --model=$MODEL_PATH --dataset=$DATASET_PATH > $MODEL"_"$DATASET"_block"$BLOCKSIZE"_"$PREEMPT_MODE".out" 2>&1
```

