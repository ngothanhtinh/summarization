FROM nvidia/cuda:11.7.1-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git\
        python3-dev \
        python3-pip \
        python3-opencv \
        libglib2.0-0

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

COPY ./mains ./mains
COPY ./models ./models
COPY ./schema ./schema
COPY ./app.py .

EXPOSE 5000

CMD uvicorn --host 0.0.0.0 --port 5000 app:app