<div align="center">

  <h1>
    <img src="assets/logo.png" alt="LoPT Logo" width="52" align="center">
    LoPT: Local-Learning Post-Training
  </h1>

  <p>
    <strong>A cheaper and faster recipe for LLM post-training with controlled task-gradient reach.</strong>
  </p>

  <p>
    LoPT inserts a midpoint stop-gradient boundary into decoder-only LLMs:
    the second-half block learns from the task objective, while the first-half block
    is updated through lightweight feature reconstruction.
  </p>

  <p>
    <a href="https://arxiv.org/abs/2605.04913">
      <img src="https://img.shields.io/badge/arXiv-2605.04913-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
    </a>
    <a href="README.md">
      <img src="https://img.shields.io/badge/README-English-2563eb?style=for-the-badge" alt="English README">
    </a>
    <a href="README_zh.md">
      <img src="https://img.shields.io/badge/README-中文-16a34a?style=for-the-badge" alt="Chinese README">
    </a>
  </p>

  <p>
    <a href="#-installation">Installation</a> ·
    <a href="#-quick-start">Quick Start</a> ·
    <a href="#-method-overview">Method</a> ·
    <a href="#-training">Training</a> ·
    <a href="#-evaluation">Evaluation</a>
  </p>

</div>

---

## 🧭 Overview

**LoPT** is a lightweight post-training recipe for decoder-only language models.  
Instead of propagating task gradients through the entire transformer stack, LoPT explicitly controls **gradient reach** by splitting the model into two or more contiguous blocks.

In the default two-block setting:

- the **second-half block** receives the standard task loss;
- the **first-half block** is protected from direct task-gradient propagation;
- earlier blocks remain trainable through a **local feature-reconstruction objective**;
- hidden states are detached at block boundaries, reducing full-depth backward coupling.

After training, auxiliary reconstruction heads are discarded. The saved model remains a standard Hugging Face causal language model and can be loaded with `AutoModelForCausalLM`.

---

## ✨ Highlights

| Capability | Description |
|---|---|
| **SFT support** | End-to-end and LoPT supervised fine-tuning |
| **GRPO support** | On-policy GRPO training with Hugging Face TRL |
| **Default LoPT split** | Two-block LoPT with `--num_blocks 2` |
| **Multi-block LoPT** | Four-block LoPT with `--num_blocks 4` |
| **HF-compatible checkpoints** | Saved as standard Hugging Face causal LM models |
| **Auxiliary heads removed at inference** | No extra inference-time computation |

---

## 🗺️ Roadmap

- [x] Release training code
- [x] Release arXiv paper
- [ ] Release checkpoint models

---

## 🧠 Method Overview

Standard end-to-end post-training backpropagates the task loss through all transformer layers.  
LoPT changes the backward path while keeping the task objective unchanged.

The model is split into contiguous blocks:

<p align="center">
  <img src="assets/framework.png" alt="LoPT Framework" width="680">
</p>

The key idea is simple:

> **Task gradients adapt the task-facing layers, while earlier layers are updated locally to preserve useful representations and maintain a compatible interface.**

This makes LoPT complementary to common efficiency tools such as LoRA, gradient checkpointing, DeepSpeed/ZeRO, and pipeline parallelism.

---

## ⚙️ Installation

```bash
git clone https://github.com/HumyuShi/LoPT.git
cd LoPT

conda create -n lopt python=3.11 -y
conda activate lopt

pip install -r requirements.txt
```

If your CUDA driver does not support the default PyTorch wheel resolved by `pip`, install a matching PyTorch wheel first, then install the remaining requirements.

For example:

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.6.0+cu118
pip install -r requirements.txt
```

For multi-GPU GRPO, configure Accelerate once:

```bash
accelerate config
```

---

## 🚀 Quick Start

### SFT with LoPT

```bash
torchrun --nproc_per_node 8 train_sft.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset_preset metamathqa \
  --method ll \
  --num_blocks 2 \
  --max_seq_length 1024 \
  --per_device_train_batch_size 4 \
  --learning_rate 2e-5 \
  --lr_k1 2e-5 \
  --lambda_aux 10 \
  --gradient_checkpointing \
  --output_dir outputs/qwen25-7b-metamathqa-lopt-k2
```

### GRPO with LoPT

```bash
accelerate launch --num_processes 8 train_grpo.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --task gsm8k \
  --method ll \
  --num_blocks 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_generations 4 \
  --steps_per_generation 1 \
  --num_iterations 1 \
  --loss_type grpo \
  --learning_rate 1e-6 \
  --lr_k1 1e-6 \
  --lambda_aux 10 \
  --gradient_checkpointing \
  --output_dir outputs/qwen25-7b-gsm8k-grpo-lopt-k2
```

---

## 🤖 Supported Models

You can pass any local path or Hugging Face model ID for Llama/Qwen/Mistral-like decoder-only models exposed as `model.layers` in `AutoModelForCausalLM`.

Built-in aliases:

| Alias | Hugging Face model |
|---|---|
| `qwen3-4b` | `Qwen/Qwen3-4B` |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `qwen2.5-32b` | `Qwen/Qwen2.5-32B-Instruct` |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` |

---

## 📚 Training Datasets

| Preset | Use | Official source |
|---|---:|---|
| `metamathqa` | SFT | https://huggingface.co/datasets/meta-math/MetaMathQA |
| `alpaca` | SFT | https://huggingface.co/datasets/tatsu-lab/alpaca |
| `tulu3` | SFT | https://huggingface.co/datasets/allenai/tulu-3-sft-mixture |
| `gsm8k` | GRPO | https://huggingface.co/datasets/openai/gsm8k |
| `numinamath` | GRPO | https://huggingface.co/datasets/AI-MO/NuminaMath-CoT |

> [!NOTE]
> Check the original model and dataset licenses before training. Some sources, including Llama-family models and Alpaca, have access or usage restrictions.

The training scripts can load directly from Hugging Face. You can also download datasets and optionally export local Parquet files:

```bash
python scripts/download_datasets.py --dataset all
python scripts/download_datasets.py --dataset metamathqa --write_parquet --output_dir data
```

Local Parquet or JSON files can be passed with:

```bash
--train_file path/to/data.parquet
```

---

## 📊 Benchmark Datasets

For benchmark reporting, we use the official dataset releases below.

| Benchmark | Area | Config / split | Official source |
|---|---|---|---|
| MMLU | multitask knowledge | official subject splits | https://huggingface.co/datasets/cais/mmlu |
| IFEval | instruction following | default release | https://huggingface.co/datasets/google/IFEval |
| ARC-Challenge | science reasoning | `ARC-Challenge` | https://huggingface.co/datasets/allenai/ai2_arc |
| GSM8K | grade-school math reasoning | `main`, `test` | https://huggingface.co/datasets/openai/gsm8k |
| HellaSwag | commonsense completion | default release | https://huggingface.co/datasets/allenai/hellaswag |
| TruthfulQA | truthfulness | multiple-choice / generation configs | https://huggingface.co/datasets/truthfulqa/truthful_qa |
| WinoGrande | commonsense coreference | validation / test release | https://huggingface.co/datasets/allenai/winogrande |

---

## 🏋️ Training

### SFT: LoPT, `k=2`

```bash
torchrun --nproc_per_node 8 train_sft.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset_preset metamathqa \
  --method ll \
  --num_blocks 2 \
  --max_seq_length 1024 \
  --per_device_train_batch_size 4 \
  --learning_rate 2e-5 \
  --lr_k1 2e-5 \
  --lambda_aux 10 \
  --gradient_checkpointing \
  --output_dir outputs/qwen25-7b-metamathqa-lopt-k2
```

### SFT: LoPT, `k=4`

```bash
torchrun --nproc_per_node 8 train_sft.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset_preset metamathqa \
  --method ll \
  --num_blocks 4 \
  --max_seq_length 1024 \
  --per_device_train_batch_size 4 \
  --learning_rate 2e-5 \
  --lr_k1 2e-5 \
  --lambda_aux 10 \
  --gradient_checkpointing \
  --output_dir outputs/qwen25-7b-metamathqa-lopt-k4
```

### SFT: E2E baseline

```bash
torchrun --nproc_per_node 8 train_sft.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset_preset metamathqa \
  --method e2e \
  --max_seq_length 1024 \
  --per_device_train_batch_size 4 \
  --learning_rate 2e-5 \
  --gradient_checkpointing \
  --output_dir outputs/qwen25-7b-metamathqa-e2e
```

### SFT: Alpaca

```bash
torchrun --nproc_per_node 8 train_sft.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset_preset alpaca \
  --method ll \
  --num_blocks 2 \
  --max_seq_length 512 \
  --per_device_train_batch_size 4 \
  --learning_rate 2e-5 \
  --lambda_aux 10 \
  --gradient_checkpointing \
  --output_dir outputs/qwen25-7b-alpaca-lopt-k2
```

### SFT: Tulu-3

```bash
torchrun --nproc_per_node 8 train_sft.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --dataset_preset tulu3 \
  --method ll \
  --num_blocks 2 \
  --max_seq_length 2048 \
  --per_device_train_batch_size 2 \
  --learning_rate 2e-5 \
  --lambda_aux 10 \
  --gradient_checkpointing \
  --output_dir outputs/llama31-8b-tulu3-lopt-k2
```

---

## 🎯 GRPO Training

GRPO uses TRL's `GRPOTrainer`, so completions are sampled from the current actor during training.

### GRPO: GSM8K, LoPT `k=2`

```bash
accelerate launch --num_processes 8 train_grpo.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --task gsm8k \
  --method ll \
  --num_blocks 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_generations 4 \
  --steps_per_generation 1 \
  --num_iterations 1 \
  --loss_type grpo \
  --learning_rate 1e-6 \
  --lr_k1 1e-6 \
  --lambda_aux 10 \
  --gradient_checkpointing \
  --output_dir outputs/qwen25-7b-gsm8k-grpo-lopt-k2
```

### GRPO: GSM8K, LoPT `k=4`

```bash
accelerate launch --num_processes 8 train_grpo.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --task gsm8k \
  --method ll \
  --num_blocks 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_generations 4 \
  --steps_per_generation 1 \
  --num_iterations 1 \
  --loss_type grpo \
  --learning_rate 1e-6 \
  --lr_k1 1e-6 \
  --lambda_aux 10 \
  --gradient_checkpointing \
  --output_dir outputs/qwen25-7b-gsm8k-grpo-lopt-k4
```

### GRPO: GSM8K, E2E baseline

```bash
accelerate launch --num_processes 8 train_grpo.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --task gsm8k \
  --method e2e \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_generations 4 \
  --steps_per_generation 1 \
  --num_iterations 1 \
  --loss_type grpo \
  --learning_rate 1e-6 \
  --gradient_checkpointing \
  --output_dir outputs/qwen25-7b-gsm8k-grpo-e2e
```

### GRPO: NuminaMath

```bash
accelerate launch --num_processes 8 train_grpo.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --task numinamath \
  --method ll \
  --num_blocks 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_generations 4 \
  --max_completion_length 512 \
  --steps_per_generation 1 \
  --num_iterations 1 \
  --loss_type grpo \
  --learning_rate 1e-6 \
  --lambda_aux 10 \
  --gradient_checkpointing \
  --output_dir outputs/qwen25-7b-numinamath-grpo-lopt-k2
```

---

## 🧪 Smoke Tests

Before launching full training, run small smoke tests with a lightweight model.

### SFT smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python train_sft.py \
  --model_name_or_path HuggingFaceTB/SmolLM2-135M-Instruct \
  --dataset_preset alpaca \
  --method ll \
  --num_blocks 2 \
  --max_seq_length 64 \
  --max_samples 16 \
  --max_steps 2 \
  --per_device_train_batch_size 1 \
  --output_dir outputs/smoke-sft-lopt
```

### GRPO smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python train_grpo.py \
  --model_name_or_path HuggingFaceTB/SmolLM2-135M-Instruct \
  --task gsm8k \
  --method ll \
  --num_blocks 2 \
  --max_samples 8 \
  --max_steps 1 \
  --num_generations 2 \
  --gradient_accumulation_steps 2 \
  --max_prompt_length 128 \
  --max_completion_length 64 \
  --steps_per_generation 2 \
  --num_iterations 1 \
  --loss_type grpo \
  --per_device_train_batch_size 1 \
  --output_dir outputs/smoke-grpo-lopt
```

---

## 📄 Paper

If you find this repository useful, please consider reading or citing the paper:

- arXiv: https://arxiv.org/abs/2605.04913

```bibtex
@misc{shi2026lopt,
  title        = {Rethinking Local Learning: A Cheaper and Faster Recipe for LLM Post-Training},
  author       = {Shi, Hengyu and Han, Tianyang and Wang, Peizhe and Wang, Zhiling and Yang, Xu and Su, Junhao},
  year         = {2026},
  eprint       = {2605.04913},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}
```

---

## 📌 Notes

- Auxiliary reconstruction heads are used only during training and are removed at inference.
- LoPT checkpoints can be loaded as standard Hugging Face causal LM models.
- `--method ll` enables LoPT-style local learning.
- `--method e2e` enables the end-to-end baseline.
- `--num_blocks 2` is the default midpoint split.
- `--num_blocks 4` enables a more fine-grained local-learning variant.

---

## 📜 License

Please check the licenses of all upstream models and datasets before training or redistribution. This repository follows the license specified in the project.
