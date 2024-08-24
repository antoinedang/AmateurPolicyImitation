FROM ubuntu:22.04

# Install all needed deps
RUN apt-get update
RUN apt-get install -y python3 python3-pip python-is-python3
RUN apt-get install -y --no-install-recommends git
RUN apt-get install libgl1-mesa-glx ffmpeg libsm6 libxext6  -y
RUN apt-get install -y vim tmux wget unzip

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN wget https://www.roboti.us/file/mjkey.txt -O /root/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}

RUN pip3 install --no-cache-dir --upgrade numpy
RUN pip3 install --no-cache-dir numpy-quaternion
RUN pip3 install opencv-python
RUN pip3 install matplotlib
RUN pip3 install mujoco-py
RUN pip3 install mujoco
RUN pip3 install gymnasium gymnasium[box2d]
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install nvidia-cudnn-cu11==8.9.6.50
RUN pip3 install stable-baselines3[extra]
RUN pip3 install tensorflow
RUN pip3 install rl_zoo3
RUN pip3 install scipy

# install dependencies for rendering with OpenCV
RUN apt-get install -y libglfw3
RUN apt-get install -y libglfw3-dev

RUN apt-get install nvidia-cuda-toolkit; exit 0
RUN apt --fix-broken install; exit 0

RUN mkdir -p /root/AmateurPolicyImitation
WORKDIR /root/AmateurPolicyImitation

COPY . /root/AmateurPolicyImitation/ 

RUN pip install -e .


RUN apt-get autoremove -y
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN rm -rf /root/.cache/pip