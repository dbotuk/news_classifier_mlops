import pandas as pd
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from os import environ
from utils import transformations


default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta (minutes=2)
}

cnn_scrapper_url = environ.get('CNN_SCRAPPER_URL')
db_server_url = environ.get('DB_SERVER_URL')


def scrap_cnn(ti):
    response = requests.get(cnn_scrapper_url + '/extract_news')
    ti.xcom_push(key='data_cnn', value=response.json()['data'])


def transform(ti):
    data_cnn = pd.read_json(ti.xcom_pull(key='data_cnn'))
    # data_2 = pd.read_json(ti.xcom_pull(key='data_2'))
    # data_3 = pd.read_json(ti.xcom_pull(key='data_3'))
    # data = pd.concat([data_cnn, data_2, data_3], axis=0)
    data_to_transform = data_cnn

    data_to_transform['text'] = data_to_transform['text'].apply(transformations.remove_tags)
    data_to_transform['text'] = data_to_transform['text'].apply(transformations.special_char)
    data_to_transform['text'] = data_to_transform['text'].apply(transformations.convert_lower)
    data_to_transform['text'] = data_to_transform['text'].apply(transformations.remove_stopwords)
    data_to_transform['text'] = data_to_transform['text'].apply(transformations.lemmatize_word)

    ti.xcom_push(key='transformed_data', value=data_to_transform.to_json())


def save_into_db(ti):
    transformed_data = ti.xcom_pull(key='transformed_data')
    requests.post(db_server_url + '/add_all_news', json={'data': transformed_data})


with DAG(
    dag_id='get_new_data',
    default_args=default_args,
    description='Extracts new data from the News Websites and Loads to DB',
    start_date=datetime(2024, 6, 7, 2),
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
