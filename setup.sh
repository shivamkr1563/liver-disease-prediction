#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y libgomp1 libglib2.0-0

# Install Python dependencies
pip install --upgrade pip
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements_streamlit.txt