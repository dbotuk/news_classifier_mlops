FROM python:3.11-slim-bookworm

WORKDIR /data_transformer

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ADD . /data_transformer

EXPOSE ${DATA_TRANSFORMER_PORT}

ENTRYPOINT [ "python" ] 
CMD ["-u", "data_transformer.py"]