import pandas as pd
import requests
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from os import environ


default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta (minutes=2)
}

db_server_url = environ.get('DB_SERVER_URL')
data_transformer_url = environ.get('DATA_TRANSFORMER_URL')


def extract_data_from_csv(ti):
    csv_file_path = os.path.join(os.path.dirname(__file__), 'init_dataset.csv')
    data = pd.read_csv(csv_file_path)
    ti.xcom_push(key='init_data', value=data.to_json())


def transform(ti):
    data = pd.read_json(ti.xcom_pull(key='init_data'))
    response = requests.post(data_transformer_url + '/transform', json={'data': data.to_json(), 'column': 'text'})
    transformed_data = response.json()['data']
    ti.xcom_push(key='transformed_data', value=transformed_data)


def save_into_db(ti):
    init_data = ti.xcom_pull(key='transformed_data')
    requests.post(db_server_url + '/add_all_news', json={'data': init_data})


with DAG(
    dag_id='load_init_data',
    default_args=default_args,
    description='Extracts init data from csv file and saves into DB',
    start_date=datetime(2024, 5, 24,2)
) as dag:
    extract_data_from_csv = PythonOperator(
        task_id='extract_data_from_csv',
        python_callable=extract_data_from_csv
    )
    transform = PythonOperator(
        task_id='transform',
        python_callable=transform
    )
    save_into_db = PythonOperator(
        task_id='save_into_db',
        python_callable=save_into_db
    )

    extract_data_from_csv >> transform >> save_into_db
