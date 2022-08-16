# For more information, please refer to https://aka.ms/vscode-docker-python
FROM continuumio/miniconda3

ARG QIIME2_RELEASE

ENV PATH /opt/conda/envs/qiime2-${QIIME2_RELEASE}/bin:$PATH
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV MPLBACKEND agg
ENV HOME /home/qiime2
ENV XDG_CONFIG_HOME /home/qiime2

RUN mkdir /home/qiime2
WORKDIR /home/qiime2

RUN curl -fsSLO https://get.docker.com/builds/Linux/x86_64/docker-17.04.0-ce.tgz \
  && tar xzvf docker-17.04.0-ce.tgz \
  && mv docker/docker /usr/local/bin \
  && rm -r docker docker-17.04.0-ce.tgz
RUN conda update -q -y conda
RUN conda install -q -y wget
RUN apt-get install -y procps
RUN wget https://data.qiime2.org/distro/core/qiime2-${QIIME2_RELEASE}-py38-linux-conda.yml
RUN conda env create -n qiime2-${QIIME2_RELEASE} --file qiime2-${QIIME2_RELEASE}-py38-linux-conda.yml
RUN rm qiime2-${QIIME2_RELEASE}-py38-linux-conda.yml
