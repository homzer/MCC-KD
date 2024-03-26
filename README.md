# MCC-KD: Multi-CoT Consistent Knowledge Distillation

<div align="center">
Hongzhan Chen<sup>1</sup>, Siyue Wu<sup>1</sup>, Xiaojun Quan<sup>1*</sup>, Rui Wang, Ming Yan<sup>2</sup>, Ji Zhang<sup>2</sup>
</div>
<div align="center">
chenhzh59@mail2.sysu.edu.cn, wusy39@mail2.sysu.edu.cn, quanxj3@mail.sysu.edu.cn
</div>
<div align="center">
<sup>1</sup>Sun Yat-sen University <sup>2</sup>Alibaba Group
</div>
<div align="center">
*Corresponding authors
</div>


<div align="center">
    <a href="https://arxiv.org/pdf/2310.14747.pdf"><img src="assets/Paper-Arxiv-orange.svg" ></a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FX-PLUG%2FMulti-LLM-Agent&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</div>

## Framework Overview

The framework applied by MCC-KD is an efficient and easy-to-develop LLM training + inference framework. This project is developed based on PyTorch and FairScale, employing tensor (model) parallelism strategy.

- Efficient Training
- Efficient Inference

The maximum supported inference batch size is **`384`**, and the maximum supported training batch size is **`4`**. This is based on using 8xV100 32GB GPUs, a model with 7 billion parameters, and a maximum sequence length of `1024`.

Compared to the HuggingFace framework, LLaMA-RLHF achieves an increase in both inference and training speed of over `2` times.

## Requirement

| Library       | Recommend | 
|---------------|-----------|
| python        | 3.8       | 
| torch         | 2.0.1    | 
| transformers | 4.37.2    | 
| fire      | 0.5.0    | 
| fairscale    | 0.4.13    | 
| sentencepiece | 0.1.97     | 
| safetensors           | 0.4.1    | 

## Current Supported Models

| Supported Models|
|---------------|
|llama-1-7b|
|llama-1-13b|
|llama-1-33b|
|llama-2-7b|
|llama-2-13b|
|llama-2-70b|
|mistral-7b-instruct-v0.2|
|mixtral-8x7b-instruct-v0.1|
|qwen-7b|
|qwen-14b|
|qwen-72b|

## Teacher Rationales

We provide training sets, validation sets, test sets, and extracted **raw** teacher rationales for datasets including GSM8K, CSQA, SVAMP, and ASDiv in the `data` directory.

## Checkpoint Downloading

The original llama version can be downloaded from https://github.com/facebookresearch/llama, which can be perfectly loaded into our framework.

Theoretically, the current model architecture can also be compatible with the model weight parameters available on Hugging Face, but further renaming of the module names is required to be able to load them. We have provided the relevant renaming functions in the `src/checkpoint.py` file. This will take a little bit of your time to make the modifications.

## Getting Started

### 1. Checkpoint Splitting

To conduct model parallel training and inference, we need to split the model checkpoint file into several parts. For example, for `world_size=8`, which means we need to split the checkpoint into 8 parts. 
Considering a model parameter file `/path/to/your/checkpoint.bin` (suffixes such as .pth, .safetensors are supported, in fact, as long as the file is stored in the form of a dictionary), run:

```shell script
torchrun checkpoint_split.py \
--ckpt_file /path/to/your/checkpoint.bin \
--save_path /path/to/save/ \
--n 8
```

You are expected to get following checkpoint files:

```
/path/to/save/consolidated.00.pth
/path/to/save/consolidated.01.pth
/path/to/save/consolidated.02.pth
/path/to/save/consolidated.03.pth
/path/to/save/consolidated.04.pth
/path/to/save/consolidated.05.pth
/path/to/save/consolidated.06.pth
/path/to/save/consolidated.07.pth
```

### 2. Model Training

Take Llama-1-7b as an example, with `lora_rank=128`, run the following script to train the model on 8 GPUs (The current settings are compatible with 8xV100 32GB.):

```shell script
torchrun --nproc_per_node 8 train.py \
--task GSM8K \
--ckpt_dir /path/to/your/ckpt/ \
--save_dir /path/to/save/ \
--train_file data/GSM8K/train.json \
--label_file data/GSM8K/test.json \
--model_type lora-llama-1-7b \
--max_batch_size 6 \
--lora_rank 128 \
--eval_batch_size 384 \
--epochs 24 \
--use_float16 True
```

If you don't want to use LoRA, change `model_type` to `llama-1-7b` and set `lora_rank=-1`.
If you want to use bfloat16 instead, replace `--use_float16=True` with `--use_bfloat16=True`. It is default to use float32, when `--use_float16=False` and `--use_bfloat16=False`.


### 3. MCC-KD Training

```shell script
torchrun --nproc_per_node 8 train_mcc.py \
--task GSM8K \
--ckpt_dir /path/to/your/ckpt/ \
--save_dir /path/to/save/ \
--train_file data/GSM8K/train-multi-cots-preview.json \
--label_file data/GSM8K/test.json \
--model_type lora-llama-1-7b \
--max_batch_size 6 \
--lora_rank 128 \
--eval_batch_size 384 \
--epochs 24 \
--use_float16 True
```

MCC-KD requires ensuring the diversity of rationales and finding a common answer span.
Make sure to include "indices" to record the starting and ending indices of the common answer span (after tokenized). It should look something like the following:

```json
[
  {
    "instruction": "...",
    "output": [
        "rationale1",
        "rationale2"
    ],
    "label": "...",
    "indices": [
        [
            23,
            28
        ],
        [
            42,
            47
        ]
    ]
  }  
]
```

We provide a preview version JSON file at `data/GSM8K/train-multi-cots-preview.json`, which typically contains fewer samples than `data/GSM8K/train.json` due to the correctness filtering.

## Citation
```
@misc{chen2024roleinteract,
      title={RoleInteract: Evaluating the Social Interaction of Role-Playing Agents}, 
      author={Hongzhan Chen and Hehong Chen and Ming Yan and Wenshen Xu and Xing Gao and Weizhou Shen and Xiaojun Quan and Chenliang Li and Ji Zhang and Fei Huang and Jingren Zhou},
      year={2024},
      eprint={2403.13679},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


