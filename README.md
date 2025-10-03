# Discrete Diffusion LLM Distillation Methods 

This repository is an extension and reproduction of several recent and representative methods for distilling Large Language Models (LLMs) using **discrete diffusion frameworks**. 

### `environment.yml`

```yaml
name: duo
channels:
  - nvidia/label/cuda-12.4.0
  - conda-forge
dependencies:
  - python=3.12.11
  - pip
  - cuda-toolkit=12.4.0
  - pip:
    - transformers==4.38.2
    - datasets==2.15.0
    - torch==2.3.1
    - torchvision==0.18.1
    - torchaudio==2.3.1
    - flash-attn==2.7.4.post1
    - einops==0.7.0
    - wandb==0.21.0
    - tqdm==4.67.1
    - lightning==2.2.1
    - triton==2.2.0
```

-----


## 📋 Implemented Distillation Methods

### 1. Progressive Distillation (SDTT)
- Implements multi-round progressive distillation in discrete diffusion models, following [Ye et al., 2025](https://arxiv.org/abs/2405.16919).
- Uses masked DiT class for text-based tasks.
- Supports extended context length and aggressive reduction in inference steps.

### 2. DiMO (Consistency)
- Implements token-level on-policy consistency distillation as described in [Zhu et al., 2025](https://arxiv.org/abs/2501.00000).
- Uses auxiliary student and teacher models with pseudo-intermediate states for text-to-image tasks.
- Utilizes a masked DiT class and long-context modeling.

### 3. DUO (Diffusion Duality)
- Reproduces the DUO framework for discrete consistency distillation and curriculum learning, described in [Sahoo et al., 2025](https://arxiv.org/abs/2506.10892).
- Supports training from scratch based on USDM and uniform DiT class.

---

## 🔬 Distillation Method Summary

| Title | Beyond Autoregression (SDTT) | DiMO | Diffusion Duality (DUO) |
| :--- | :--- | :--- | :--- |
| **Distill method** | Progressive  | On-Policy  | Consistency  |
| **\# distill rounds** | 7 | 1 | 5 |
| **Base Model** | MDLM | Meissonic | Trained from scratch based on USDM |
| **DiT class** | MDM  | MDM  | USDM  |
| **Tasks** | Text | Text to Image | Text |
| **Context Length** | 1024 | 4096 | 1024 |
| **inference steps** | 1024 → 16‑32 | 32 → 1 | 1024 → 8‑16 |
| **Details** | 1/2 distill, multiple rounds | student generation from $x_{\text{init}}$ to $x_\theta$<br>forward from $x_\theta$ to $x_t$<br>backward from $x_t$ to $x_0$ via $p_\phi$ and $p_\psi$  | Curriculum learning + <br> Discrete Consistency Distillation  |

---

## 🏗️ How to Use



1.  **Clone the repository:**

    ```bash
    git clone git@github.com:haoyangzheng-ai/discrete_diffusion.git
    cd discrete_diffusion
    ```

2.  **Create and activate the Conda environment:**
    The environment name specified in the file is `duo`.

    ```bash
    conda env create -f environment.yml
    conda activate duo
    ```

3.  **Run the desired distillation method:**

      - **SDTT:**
        ```bash
        source ./script/distill_sdtt_openwebtext.sh
        ```
      - **DiMO:**
        ```bash
        source ./script/distill_dimo_openwebtext.sh
        ```
      - **DUO:**
        ```bash
        source ./script/distill_duo_openwebtext.sh
        ```

4.  **Evaluate the distilled models:**

      - **SDTT:**
        ```bash
        source ./script/eval_distill_sdtt_openwebtext.sh
        ```
      - **DiMO:**
        ```bash
        source ./script/eval_distill_dimo_openwebtext.sh
        ```
      - **DUO:**
        ```bash
        source ./script/eval_distill_duo_openwebtext.sh
        ```

---

## 📚 References

-   **DUO**: ["Diffusion Duality: Curriculum and Consistency for Discrete Diffusion LLMs"](https://arxiv.org/abs/2506.10892). ICLR 2025.
-   **SDTT**: ["Beyond Autoregression: Progressive Distillation for Discrete Diffusion Language Models"](https://arxiv.org/abs/2405.16919). ICLR 2025.
-   **DiMO**: ["DiMO: On-Policy Consistency Distillation for Discrete Diffusion"](https://arxiv.org/abs/2501.00000). ICCV 2025.
-   **MDLM**: ["Simple and Effective Masked Diffusion Language Models"](https://arxiv.org/abs/2406.07524). NeurIPS 2024.
-   ["Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis"](https://arxiv.org/abs/2410.08261). ICLR 2025.
-   ["Progressive Distillation for Fast Sampling of Diffusion Models"](https://arxiv.org/abs/2202.00512). ICLR 2022.
-   ["On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"](https://arxiv.org/abs/2306.13649). ICLR 2024.
-   ["Consistency Models"](https://arxiv.org/abs/2303.01469). ICML 2023.
---

## ✨ Acknowledgements

This repository builds on [DUO](https://github.com/s-sahoo/duo). The diffusion model implementations are heavily inspired by the following papers and their official codebases:

  - **DUO**: [Diffusion Duality: Curriculum and Consistency for Discrete Diffusion LLMs](https://github.com/s-sahoo/duo) by Subham Sahoo.
  - **SDTT**: [Beyond Autoregression: Progressive Distillation for Discrete Diffusion Language Models](https://github.com/jdeschena/sdtt) by Justin Deschenaux.
  - **DiMO**: [DiMO: On-Policy Consistency Distillation for Discrete Diffusion](https://github.com/yuanzhi-zhu/DiMO) by Yuanzhi Zhu.

