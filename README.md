<h2 align="center"><span style="color: #ff6b35; font-weight: bold; font-size: 1.8em;"> Martian World Model</span><br>
  <span style="font-size: 1.6em; color: #666;">Controllable Video Synthesis with Physically Accurate 3D Reconstructions</span>

<a href="https://arxiv.org/abs/2507.07978" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-martian--world--model-red?logo=arxiv" height="20" />
</a>
<a href="https://marsgenai.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/Website-martian--world--model-blue.svg" height="20" />
</a>
<a href="https://huggingface.co/datasets/LongfeiLi/M3arsSynth">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Dataset-F0CD4B?labelColor=666EEE" alt='HuggingFace Dataset'>
</a>



## Introduction

Martian World Model introduces a comprehensive solution for synthesizing realistic Martian landscape videos, addressing the critical need for mission rehearsal and robotic simulation in planetary exploration.




## Installation

To set up the environment for Martian World Model, follow these steps:

### 1. Download Model Weights

**VGGT-1B**:
```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf download facebook/VGGT-1B --local-dir ./facebook/VGGT-1B
```

**GIM**: Download from [Google Drive](https://drive.google.com/file/d/1gk97V4IROnR1Nprq10W9NCFUv2mxXR_-/view?usp=sharing) and place in `gim/weights/`.

> **Note:** The Metric3D model (`metric3d_vit_giant2`, ~5 GB) is downloaded automatically via `torch.hub` on first run and cached in `~/.cache/torch/hub/checkpoints/`.

### 2. Build and Run Docker Container

```bash
# Set your GitHub token (required for Metric3D download via torch.hub)
export GITHUB_TOKEN=ghp_your_token_here

# Build the Docker image
bash docker/docker_build.sh mars

# Run the container (mounts project directory, passes GITHUB_TOKEN)
bash docker/docker_run.sh mars
```

## Usage

To use the M3arsSynth for synthesizing Martian landscape videos, follow these steps:

1. Replace the `DATA_ROOT_DIR` variable in `scripts/infer.sh` with the absolute path to your project directory.

2. Run the following command:

```bash
bash scripts/infer.sh
```

This will execute the pipeline for co-visible global geometry initialization, pose optimization, and video rendering.
