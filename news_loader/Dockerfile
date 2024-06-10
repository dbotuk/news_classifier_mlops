FROM apache/airflow:2.0.1

USER root

WORKDIR /news_loader

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ADD . /news_loader

USER airflow