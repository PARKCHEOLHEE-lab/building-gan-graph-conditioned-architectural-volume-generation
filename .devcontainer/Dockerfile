FROM nvcr.io/nvidia/pytorch:23.10-py3

COPY requirements.txt /requirements.txt
COPY post-create-command.sh /post-create-command.sh

RUN python -m pip install --upgrade pip
RUN pip install -r /requirements.txt

RUN apt-get update && apt-get install -y zsh libsm6 libxext6 libxrender-dev libgl1-mesa-glx xvfb
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
