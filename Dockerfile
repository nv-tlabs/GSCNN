FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get update \
    && apt-get install -y software-properties-common  wget \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y make git curl vim vim-gnome 

# Install apt-get
RUN apt-get install -y python3-pip python3-dev vim htop python3-tk pkg-config 

RUN pip3 install --upgrade pip==9.0.1

# Install from pip
RUN pip3 install pyyaml \
                 scipy==1.1.0 \
                 numpy \
                 tensorflow \
                 scikit-learn \
                 scikit-image \
                 matplotlib \
                 opencv-python \
                 torch==1.0.0 \
                 torchvision==0.2.0 \
                 torch-encoding==1.0.1 \
                 tensorboardX \
                 tqdm

