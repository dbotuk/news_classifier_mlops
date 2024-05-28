FROM python:3.11-slim-bookworm

WORKDIR /scrappers/cnn_scrapper

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ADD . /scrappers/cnn_scrapper

EXPOSE ${CNN_SCRAPPER_PORT}

ENTRYPOINT [ "python" ]
CMD ["-u", "cnn_scrapper.py"]
