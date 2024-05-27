FROM python:3.11-slim-bookworm

WORKDIR /db_server

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ADD . /db_server

EXPOSE 4000

ENTRYPOINT [ "python" ] 
CMD ["-u", "db_server.py"]