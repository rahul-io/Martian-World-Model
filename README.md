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

Martian World Models introduces a comprehensive solution for synthesizing realistic Martian landscape videos, addressing the critical need for mission rehearsal and robotic simulation in planetary exploration.




## Installation

To set up the environment for Martian World Models, follow these steps:

```bash
# Create a new conda environment
conda create -n mars python=3.10.13 cmake=3.14.0 -y

# Activate the environment
conda activate mars

# Install PyTorch and CUDA dependencies
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install required Python packages
pip install -r requirements.txt

# Install submodules
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/fused-ssim

```

### Download Image Matching Module Weights

To download the GIM weights required for the project, use the following link:

[Google Drive](https://drive.google.com/file/d/1gk97V4IROnR1Nprq10W9NCFUv2mxXR_-/view?usp=sharing)

After downloading, place the weights in the `gim/weights/` directory.

## Usage

To use the M3arsSynth for synthesizing Martian landscape videos, follow these steps:

1. Replace the `DATA_ROOT_DIR` variable in `scripts/infer.sh` with the absolute path to your project directory.

2. Run the following command:

```bash
bash scripts/infer.sh
```

This will execute the pipeline for co-visible global geometry initialization, pose optimization, and video rendering.
