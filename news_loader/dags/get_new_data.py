from datetime import datetime, timedelta
from airflow.operators.bash import BashOperator
from airflow import DAG

default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta (minutes=2)
}

with DAG(
    dag_id='scrap_new_data',
    default_args=default_args,
    description='test',
    start_date=datetime(2024, 5, 1, 2),
    schedule_interval='@daily'
) as dag:
    task1 = BashOperator(
        task_id='task1',
        bash_command="echo hello world task1"
    )
    task2 = BashOperator(
        task_id='task2',
        bash_command="echo hello world task2"
    )
    task3 = BashOperator(
        task_id='task3',
        bash_command="echo hello world task3"
    )

    task1 >> [task2, task3]