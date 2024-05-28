import pandas as pd
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from os import environ

default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta (minutes=2)
}

cnn_scrapper_url = environ.get('CNN_SCRAPPER_URL')
data_transformer_url = environ.get('DATA_TRANSFORMER_URL')
db_server_url = environ.get('DB_SERVER_URL')


def scrap_cnn(ti):
    response = requests.get(cnn_scrapper_url + '/extract_news')
    ti.xcom_push(key='data_cnn', value=response.json()['data'])


def transform(ti):
    data_cnn = pd.read_json(ti.xcom_pull(key='data_cnn'))
    # data_2 = pd.read_json(ti.xcom_pull(key='data_2'))
    # data_3 = pd.read_json(ti.xcom_pull(key='data_3'))
    # data = pd.concat([data_cnn, data_2, data_3], axis=0)
    data = data_cnn
    response = requests.post(data_transformer_url + '/transform', json={'data': data.to_json(), 'column': 'text'})
    transformed_data = response.json()['data']
    ti.xcom_push(key='transformed_data', value=transformed_data)


def save_into_db(ti):
    transformed_data = ti.xcom_pull(key='transformed_data')
    requests.post(db_server_url + '/add_all_news', json={'data': transformed_data})


with DAG(
    dag_id='get_new_data',
    default_args=default_args,
    description='Extracts new data from the News Websites and Loads to DB',
    start_date=datetime(2024, 5, 24,2),
    schedule_interval='@daily'
) as dag:
    scrap_cnn = PythonOperator(
        task_id='scrap_cnn',
        python_callable=scrap_cnn
    )
    transform = PythonOperator(
        task_id='transform',
        python_callable=transform
    )
    save_into_db = PythonOperator(
        task_id='save_into_db',
        python_callable=save_into_db
    )

    scrap_cnn >> transform >> save_into_db
