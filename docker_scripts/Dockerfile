FROM nvcr.io/nvidia/pytorch:21.08-py3


# install system
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --fix-missing -y \
    build-essential \
    clang-format \
    curl \
    gdb \
    libxcb-xkb1 \
    gfortran \
    git \
    git-lfs \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libfontconfig1 \
    libgtk-3-dev \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libswscale-dev \
    libtiff-dev \
    libv4l-dev \
    libx264-dev \
    libxext6 \
    libxrender1 \
    libxvidcore-dev \
    pkg-config \
    python3-pip \
    swig \
    time \
    unzip \
    vim \
    wget \
    yasm \
    python3-dev \
    locales \
    zsh \
    strace \
    silversearcher-ag \
    sudo \
    ninja-build \
    bash-completion \
    ffmpeg \
    valgrind \
    && \
    rm -rf /var/lib/apt/lists/* \
    && \
    locale-gen "en_US.UTF-8"


WORKDIR /root
RUN wget https://code-server.dev/install.sh && \
    sh install.sh --method=standalone && \
    wget https://github.com/microsoft/vscode-cpptools/releases/download/1.4.1/cpptools-linux.vsix && \
    /root/.local/bin/code-server --install-extension cpptools-linux.vsix && \
    /root/.local/bin/code-server --install-extension twxs.cmake && \
    /root/.local/bin/code-server --install-extension ms-vscode.cmake-tools && \
    /root/.local/bin/code-server --install-extension xaver.clang-format && \
    /root/.local/bin/code-server --install-extension ms-python.python && \
    /root/.local/bin/code-server --install-extension mhutchie.git-graph && \
    /root/.local/bin/code-server --install-extension christian-kohler.path-intellisense


# install python
RUN pip3 install opencv-python==4.5.3.56 --no-cache-dir
COPY requirements.txt .
RUN pip3 install -r requirements.txt --no-cache-dir

WORKDIR /

