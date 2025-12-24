# Cantonese Soccer Commentary & Talking Head Video Generation

This repository demonstrates an AI-powered system that generates soccer commentary with synchronized talking head videos using advanced speech synthesis and lip-sync technology.

## Demo Videos
### Video Source
The source video is extracted from [here](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.facebook.com/ChungChiKwongchannel/videos/%25E4%25BC%258D%25E6%2599%2583%25E6%25A6%25AE%25E5%25A0%25B1%25E5%25B0%258E2004-%25E6%25AD%2590%25E5%259C%258B%25E7%259B%2583-%25E6%25B3%2595%25E5%259C%258B-21%25E8%258B%25B1%25E6%25A0%25BC%25E8%2598%25AD/1905282819776195/&ved=2ahUKEwiH3uPFsaqPAxUQElkFHdenCA4QtwJ6BAgQEAI&usg=AOvVaw2sLuZ0DgyOrv5iGperOuTw)

### Pure Lip Sync Demo
https://github.com/user-attachments/assets/c24946a7-0f81-490c-840a-9f5c3c9300aa

### Commentary Results
https://github.com/user-attachments/assets/41373536-0d6b-4d08-be6a-fc39612c4176

### Full Demo Recording
https://github.com/user-attachments/assets/c7c591a9-0d32-4a28-9794-f3880621b8c5

## Features
- **AI Soccer Commentary**: Generate realistic soccer match commentary using LLM
- **Talking Head Generation**: Create synchronized talking head videos with lip-sync
- **High-Quality Audio**: Advanced TTS (Text-to-Speech) synthesis
- **Real-time Processing**: Optimized for GPU acceleration

# Requirement 
1. NVidia GPU with CUDA support (1*RTX4060 is enough)
2. Ubuntu 20.04 or higher
3. Driver version >= 570.133 
4. CUDA version >= 12.0
5. The environment must be created with Python 3.10 (CosyVoice-ttsfrd requires Python 3.10)
6. [ModelScope API](https://www.modelscope.cn/my/myaccesstoken) key is required for LLM.

# Installation 
## Git 
1. Clone the repository:
```bash
git clone https://github.com/allanchan339/VLM_Soccer_Commentator_THG --depth 1  
cd VLM_Soccer_Commentator_THG
git submodule update --init --recursive
```

## Conda
1. Install Miniconda or Anaconda, then run following commands
`conda env create -f environment_torch2.4.yml`

2. Activate the environment:
```bash
conda activate SoCommVoice2.4
```

## Additional Dependencies
### Install additional dependencies for musetalk:
```bash
# Install dependencies related to musetalk
pip install --no-cache-dir -U openmim
mim install mmengine 
```

#### Install mmdet (For Pytorch 2.4.1 only)
Then we need to install mmdet and mmpose from source code and comment out the compatibility check in init.py. Otherwise, assertion error will be raised.

```bash
cd mmdetection
# Comment out the compatibility check in init.py
nano {python_path}/lib/python3.10/site-packages/mmdet/__init__.py 
```

Change the line 17 from:
```python        
and mmcv_version < digit_version(mmcv_maximum_version)), \
```
to:
```python        
and mmcv_version <= digit_version(mmcv_maximum_version)), \
```

#### Install mmpose
```bash
mim install "mmpose>=1.1.0" # not exist in conda-forge
```

<!-- ### Install additional dependencies for CosyVoice: (Ignored as yet implemented)
```bash
# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
``` -->

<!-- ### Install additional dependencies for PaddleSpeech (Ignored):
```bash
pip install paddlespeech paddlepaddle --no-deps
pip install yacs g2p-en opencc pypinyin pypinyin-dict opencc-python-reimplemented braceexpand ToJyutping webrtcvad zhon timer
``` -->

## Download pre-trained models
<!-- ### Download the pre-trained models and install CosyVoice-ttsfrd (Ignored as not required):
```bash
# Download the CosyVoice model
python download_model_cosyvoice.py

# Install the CosyVoice-ttsfrd model (Optional, if not installed, wetext will be used)
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```
 -->
### Download the pre-trained models and install MuseTalk:
```bash
# Download the MuseTalk model
sh ./download_THG_weight.sh

# Download the GPT-SoVITS models:
bash ./download_TTS_weight.sh
```

# Run the demo
```bash
python web_ui_all.py
```
