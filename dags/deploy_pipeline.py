from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

with DAG(
    dag_id="deploy_pipeline",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
) as dag:

    build_images = BashOperator(
        task_id="build_images",
        bash_command="docker-compose build ml_api ml_chatbot ml_dashboard",
    )

    push_registry = BashOperator(
        task_id="push_registry",
        bash_command="docker-compose push ml_api ml_chatbot ml_dashboard",
    )

    build_images >> push_registry
