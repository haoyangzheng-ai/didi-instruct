# Ultra-Fast Language Generation via <br>Discrete Diffusion Divergence Instruct (DiDi-Instruct)

[![Blog](https://img.shields.io/badge/Blog-0366d6?logo=gitbook&logoColor=white)](https://haoyangzheng.github.io/research/didi-instruct/)
[![Google Drive](https://img.shields.io/badge/GoogleDrive-34a853?logo=google-drive&logoColor=white)](https://drive.google.com/drive/folders/1bQlwZoaowkGy3FXnrtb4YEleKIDHrQNE?usp=sharing)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-ff6f00?logo=huggingface&logoColor=white)](https://huggingface.co/haoyangzheng/didi-instruct-small)
[![arXiv](https://img.shields.io/badge/arXiv-2509.25035-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.25035)
[![Python](https://img.shields.io/badge/Python-3.12.11-yellow?logo=python&logoColor=white)](https://github.com/haoyangzheng-ai/didi-instruct/blob/main/environment.yml) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?logo=opensourcehardware&logoColor=white)](./LICENSE.md)

By [Haoyang Zheng](https://scholar.google.com/citations?hl=en&user=cq_f7MUAAAAJ&view_op=list_works&sortby=pubdate), [Xinyang Liu](https://xinyangatk.github.io/), [Cindy Xiangrui Kong](https://xiangruikong.com/), [Nan Jiang](https://jiangnanhugo.github.io/), [Zheyuan Hu](https://scholar.google.com/citations?user=On2YFigAAAAJ&hl=zh-CN),
[Weijian Luo](https://pkulwj1994.github.io/), [Wei Deng](https://www.weideng.org/), and [Guang Lin](https://www.math.purdue.edu/~lin491/)

---

## 🔄 Updates

* **2025-10-06**: We update the [Blog](https://haoyangzheng.github.io/research/didi-instruct/).
* **2025-10-05**: We released the checkpoint on [Hugging Face](https://huggingface.co/haoyangzheng/didi-instruct-small).
* **2025-10-03**: We updated the [evaluation code](https://github.com/haoyangzheng-ai/didi-instruct/blob/main/scripts/eval-didi-instruct.sh) and released [the model checkpoint](https://drive.google.com/drive/folders/1bQlwZoaowkGy3FXnrtb4YEleKIDHrQNE?usp=sharing).
* **2025-09-29**: We uploaded our work to [arXiv](https://arxiv.org/abs/2509.25035).

### Planned Releases

* Training code for reproduction and further research.

---

## Abstract

Fast and high-quality language generation is the holy grail that people pursue in the age of AI. In this work, we introduce **Di**screte **Di**ffusion Divergence **Instruct** (**DiDi-Instruct**), a training-based method that initializes from a pre-trained (masked) discrete diffusion language model (dLLM) and distills a few-step student for fast generation. The resulting DiDi-Instruct model achieves comparable or superior performance to its dLLM teacher and the GPT-2 baseline while enabling up to **64×** acceleration. The theoretical foundation of DiDi-Instruct is a novel framework based on integral KL-divergence minimization, which yields a practical training algorithm. We further introduce grouped reward normalization, intermediate-state matching, and the reward-guided ancestral sampler that significantly improve training stability, model coverage, and inference quality. On OpenWebText, DiDi-Instruct achieves perplexity from 62.2 (8 NFEs) to 18.4 (128 NFEs), which outperforms prior accelerated dLLMs and GPT-2 baseline. These gains come with a negligible entropy loss (around 1\%) and reduce additional training wall-clock time by more than **20×** compared to competing dLLM distillation methods. We further validate the robustness and effectiveness of DiDi-Instruct through extensive ablation studies, model scaling, and the generation of discrete protein sequences. In conclusion, DiDi-Instruct is an efficient yet effective distillation method, enabling language generation in the blink of an eye.

---

## 🚀 Feel the Generation Speed

### Auto-Regressive Model (GPT-2 Small)<br><sub>Token-by-token generation → high latency</sub>
![ARM](https://github.com/haoyangzheng-ai/didi-instruct/blob/main/demos/arm.gif)

### Masked Diffusion Model (MDLM, 169M)<br><sub>Iterative denoising → faster than GPT-2 Small.</sub>
![MDLM](https://github.com/haoyangzheng-ai/didi-instruct/blob/main/demos/mdlm.gif)

### DiDi-Instruct (distilled from 169M MDLM)<br><sub>Distilled few-step student → up to **64× speedup** with matched/better quality.</sub>
![DiDi-Instruct](https://github.com/haoyangzheng-ai/didi-instruct/blob/main/demos/didi-instruct.gif)

---

## 🏗️ Usage Guide

### 1. Create and Activate the Conda Environment

Before first use, create and activate the environment from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate mask_model
```

### ~~2. Train a Small Model~~

~~Before distillation, train a baseline small model using one of the predefined scripts. For example, with the OpenWebText dataset:~~

Please refer to [this script from DUO](https://github.com/s-sahoo/duo/blob/main/scripts/train_owt_mdlm.sh) or use the checkpoint at [Google Drive](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau) (mdlm.ckpt).

### ~~3. Distill the Model~~

We will release the distillation code in the future. 

We here provide **two options** to obtain the model checkpoint:

- **Option 1 (from Google Drive):**  
  - Download the checkpoint and place the `.ckpt` file under the folder `./out/`.  
  - See [Google Drive](https://drive.google.com/drive/folders/1bQlwZoaowkGy3FXnrtb4YEleKIDHrQNE?usp=sharing) (`didi-instruct.ckpt`).  

- **Option 2 (from Hugging Face):**  
  - We also release the model checkpoint on [Hugging Face](https://huggingface.co/haoyangzheng/didi-instruct-small).  
  - You can directly run the following command to download the model and convert it into `.ckpt` format:  

    ```bash
    python ./models/hf_to_ckpt.py --hf_repo_id "haoyangzheng/didi-instruct-small" --output_dir "/your_code_path/didi-instruct/out/didi-instruct.ckpt"
    ```

### 4. Evaluate the Distilled Model

This step assesses the quality and efficiency of the distilled student by measuring perplexity and entropy against the teacher and baseline models.

```bash
source ./script/eval-didi-instruct.sh
```

---
## 📁 Repository Structure

```
didi-instruct/
├── configs/              # Configuration files directory, including experiment parameters and hyperparameter settings
├── models/               # Model definitions and related implementation code
├── scripts/              # Training and inference scripts
├── out/                  # Save pretrained models here
├── algo.py               # Algorithm implementations (DiDi-Instruct)
├── dataloader.py         # Core data loading and preprocessing code
├── dit.py                # Diffusion Transformer implementations
├── main.py               # Main file for training and evaluation
├── metrics.py            # Evaluation metrics code
├── trainer_base.py       # Base trainer class
├── utils.py              # Utility functions
├── LICENSE.md            # License file
├── README.md             # Project documentation and usage instructions
└── environment.yml       # Dependency environment specification (conda requirements)
```
---

## 📚 References

This repository is built upon [DUO](https://github.com/s-sahoo/duo): ["The Diffusion Duality. ICML 2025"](https://arxiv.org/abs/2506.10892).

We also adopt ideas from [DiMO](https://github.com/yuanzhi-zhu/DiMO), [MDLM](https://github.com/kuleshov-group/mdlm), [SDTT](https://github.com/jdeschena/sdtt), and [nanoGPT](https://github.com/karpathy/nanoGPT).

---

## 📖 Citation

If you find this repository useful, please cite the following work:

```
@article{zheng2025ultra,
  title={{Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct}},
  author={Zheng, Haoyang and Liu, Xinyang and Kong, Cindy Xiangrui and Jiang, Nan and Hu, Zheyuan and Luo, Weijian and Deng, Wei and Lin, Guang},
  journal={arXiv preprint arXiv:2509.25035},
  year={2025}
}
```

---
