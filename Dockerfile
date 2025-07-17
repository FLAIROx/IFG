#FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04
FROM nvidia/cuda:12.5.0-runtime-ubuntu20.04

# declare the image name
# ENV IMG_NAME=11.6.0-devel-ubuntu20.04 \
    # declare what jaxlib tag to use
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly
#   JAXLIB_VERSION=0.3.2


ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install python3-pip
RUN chmod 1777 /tmp
RUN apt-get update --fix-missing
RUN apt upgrade -y
RUN apt-get install software-properties-common -y
RUN apt-get install vim tmux git --fix-missing -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN  apt install cuda-nvcc-12-6 -y

RUN  apt install python3.11 python3.11-distutils -y

RUN apt-get install curl -y
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 
RUN apt install python3.11-dev -y

# install dependencies via pip
ARG UID=3566
ARG GID=100

RUN userdel -f $(getent passwd $UID | cut -d: -f1) 2>/dev/null || true
RUN groupadd -g $GID eltayeb || true
RUN useradd -d /project -u $UID -g $GID --create-home ifg_user 
RUN mkdir /scratch
RUN mkdir /mount
RUN ln -sf $(which python3.11) /usr/bin/python


RUN chmod +777 /mount
RUN chmod +777 /scratch
USER ifg_user 

COPY  requirements.txt .
RUN mkdir /project/IFG
COPY  . /project/IFG
RUN python3.11 -m pip install -r requirements.txt
ENV DS_BUILD_OPS=0
ENV DS_BUILD_CPU_ADAM=1
RUN pip install deepspeed --no-cache
#ENV PATH "$PATH:/usr/local/bin"
ENV PATH "$PATH:/project/.local/bin"
RUN chown -R ifg_user:eltayeb /project
WORKDIR /project
CMD ["/bin/bash"]
