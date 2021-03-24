FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
RUN apt update  && apt -y upgrade
RUN apt install -y\
    cmake \
    g++ \
    python3 \
    python3-pip \
	tmux
ENV LANG C

RUN echo '%users ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers             
RUN mkdir /workspace
WORKDIR /workspace
COPY ./requiments.txt /Requiments/requiments.txt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /Requiments/requiments.txt
RUN python3 -m pip install gpustat cupy-cuda100
RUN python3 -m pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install pytorch-lightning==1.1.3 mlflow==1.14.1
