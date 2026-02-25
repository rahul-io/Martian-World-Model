FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /workspace

# Install system dependencies + Python 3.10 (default on Ubuntu 22.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ gfortran git patch wget pkg-config \
    liblapack-dev libmetis-dev curl apt-utils \
    python3.10 python3.10-dev python3-pip \
    libglib2.0-0 libgl1-mesa-glx libsm6 libxext6 libxrender-dev libglu1-mesa \
    libglm-dev \
    vim nano screen tmux htop openssh-client unzip \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Symlink python / python3 to python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Install cmake 3.14.0 from the official binary release
RUN wget -q https://cmake.org/files/v3.14/cmake-3.14.0-Linux-x86_64.sh -O /tmp/cmake.sh && \
    bash /tmp/cmake.sh --skip-license --prefix=/usr/local && \
    rm /tmp/cmake.sh

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch and CUDA dependencies
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Copy requirements and install
COPY requirements.txt /workspace/requirements.txt
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu128 && \
    pip install -r /workspace/requirements.txt

# Copy the full project into the image
COPY . /workspace/Martian-World-Model

WORKDIR /workspace/Martian-World-Model

# Install submodules
# Set TORCH_CUDA_ARCH_LIST to avoid "IndexError: list index out of range"
# This targets common NVIDIA GPU architectures (Volta, Turing, Ampere, Ada)
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0 12.0+PTX"
ENV CUDA_ARCHITECTURES="70;75;80;86;89;90;120"
RUN pip install submodules/simple-knn --no-build-isolation && \
    pip install submodules/diff-gaussian-rasterization --no-build-isolation && \
    pip install submodules/fused-ssim --no-build-isolation

# CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

CMD ["/bin/bash"]
