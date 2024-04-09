# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.6.1-cuda12.1.0

# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

COPY builder/openchat /openchat
RUN cd /openchat && python3.11 -m pip install . 

ENV HF_HOME="/runpod-volume/huggingface"
ENV HF_HUB_CACHE="/runpod-volume/huggingface"
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

ADD src .
RUN mkdir /results

CMD python3.11 -u /handler.py
