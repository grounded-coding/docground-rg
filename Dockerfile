# Use the base image from PyTorch
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt && chmod 777 /opt
RUN mkdir -p /opt/nltk_data/tokenizers /opt/nltk_data/corpora && chmod 777 /opt/nltk_data/tokenizers /opt/nltk_data/corpora

# Install required packages and libraries
RUN apt-get update && apt-get install -y zsh git curl vim wget unzip gzip tar sudo ca-certificates bzip2
 # Install JDK
RUN apt-get -y install openjdk-11-jre-headless

RUN useradd -m user && echo "user:user" | chpasswd && adduser user sudo
USER user
WORKDIR /home/user

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
ENV PATH="/usr/local/cuda/bin:$PATH"
ENV CUDA_HOME=/usr/local/cuda
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda && \
    rm miniconda.sh

ENV PATH="/opt/miniconda/bin:$PATH"
RUN pip install --upgrade pip urllib3 chardet

RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html

# Java
ENV JAVA_HOME=/usr/lib/jvm/default-java 
ENV PATH="${PATH}:${JAVA_HOME}/bin"

# Sisyphus
RUN pip3 install psutil flask ipython \
    && pip3 install git+https://github.com/rwth-i6/sisyphus

# NLTK
ENV NLTK_DATA="/opt/nltk_data" \
    NCCL_DEBUG="INFO" \
    OMP_NUM_THREADS=12

RUN wget -P /opt/nltk_data/tokenizers https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip \
    && wget -P /opt/nltk_data/corpora https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip \
    && unzip /opt/nltk_data/tokenizers/punkt.zip -d /opt/nltk_data/tokenizers \
    && unzip /opt/nltk_data/corpora/wordnet.zip -d /opt/nltk_data/corpora \
    && rm /opt/nltk_data/tokenizers/*.zip /opt/nltk_data/corpora/*.zip

# Project
RUN pip3 install \
    accelerate \
    git+https://github.com/nils-hde/LLEval.git \
    git+https://github.com/nils-hde/UniEval.git \
    datasets \
    'deepspeed>=0.9.5' \
    'nltk>=3.6.7' \
    'numpy>=1.22.0' \
    openai \
    paramiko \
    'protobuf==3.20.*' \
    peft \
    pynvml \
    'rouge_score>=0.1.2' \
    'scikit_learn>=1.1.1' \
    'sentencepiece>=0.1.96' \
    'strsimpy>=0.2.1' \
    'summ_eval>=0.892' \
    'tensorboard>=2.9.0' \
    'tensorboardX>=2.5' \
    'tqdm>=4.62.3' \
    'transformers>=4.30.*'
    
RUN python3 -c "from summ_eval.meteor_metric import MeteorMetric" \
    && mkdir -p /opt/miniconda/lib/python3.10/site-packages/summ_eval/data \
    && wget https://github.com/lichengunc/refer2/raw/master/evaluation/meteor/data/paraphrase-en.gz -O /opt/miniconda/lib/python3.10/site-packages/summ_eval/data/paraphrase-en.gz

RUN pip3 uninstall bitsandbytes && pip3 install bitsandbytes

WORKDIR /app