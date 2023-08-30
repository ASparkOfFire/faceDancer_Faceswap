FROM nvcr.io/nvidia/pytorch:22.10-py3
#FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /workspace
COPY ./src/requirements.txt /workspace 

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 tzdata -y

RUN pip install -r /workspace/requirements.txt

ENTRYPOINT ["python", "api.py"]
