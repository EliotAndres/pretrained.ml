# docker-keras - Keras in Docker with Python 3 and TensorFlow on CPU

FROM debian:stretch
MAINTAINER gw0 [http://gw.tnode.com/] <gw.2017@ena.one>

# install debian packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    build-essential \
    g++ \
    git \
    openssh-client \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-virtualenv \
    python3-wheel \
    pkg-config \
    libopenblas-base \
    python3-numpy \
    python3-scipy \
    python3-h5py \
    python3-yaml \
    protobuf-compiler \
    python-lxml \
    python3-pydot \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ARG TENSORFLOW_VERSION=1.3.0
ARG TENSORFLOW_DEVICE=cpu
ARG TENSORFLOW_APPEND=
RUN pip3 --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_DEVICE}/tensorflow${TENSORFLOW_APPEND}-${TENSORFLOW_VERSION}-cp35-cp35m-linux_x86_64.whl

ARG KERAS_VERSION=2.0.8
ENV KERAS_BACKEND=tensorflow
RUN pip3 --no-cache-dir install --no-dependencies git+https://github.com/fchollet/keras.git@${KERAS_VERSION}

# quick test and dump package lists
RUN python3 -c "import tensorflow; print(tensorflow.__version__)" \
 && dpkg-query -l > /dpkg-query-l.txt \
 && pip3 freeze > /pip3-freeze.txt

COPY ./requirements.txt /opt/srv/requirements.txt
WORKDIR /opt/srv

RUN pip3 install -r requirements.txt

RUN adduser --disabled-password --gecos '' celery-user

WORKDIR /opt/srv/tensorflow_models
