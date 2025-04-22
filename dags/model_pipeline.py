from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

with DAG(
    dag_id="model_pipeline",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
) as dag:

    train = BashOperator(
        task_id="train_models",
        bash_command="python /opt/airflow/scripts/train.py",
    )

    validate = BashOperator(
        task_id="validate_models",
        bash_command="python /opt/airflow/scripts/validate_model.py",
    )

    train >> validate
