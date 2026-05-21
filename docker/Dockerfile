FROM python:3.13-slim

RUN groupadd --gid 1000 service && \
    useradd --uid 1000 --gid service --shell /bin/bash --create-home service

WORKDIR /home/service

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./src ./src
COPY ./models ./models
COPY ./data ./data

USER service

WORKDIR /home/service/src

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]