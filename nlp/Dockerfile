FROM nvcr.io/nvidia/pytorch:25.11-py3

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt update && apt install -y cuda-compat-13-0
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /workspace

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models ./models
COPY src .

CMD uvicorn nlp_server:app --port 5004 --host 0.0.0.0
