# env
pip install -r requirements.txt
apt-get install -y libglib2.0-0
apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglu1-mesa


HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download THUDM/CogVideoX-5b-I2V --local-dir THUDM/CogVideoX-5b-I2V