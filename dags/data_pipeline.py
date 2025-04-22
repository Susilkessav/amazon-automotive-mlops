from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

with DAG(
    dag_id="data_pipeline",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
) as dag:

    ingest_meta = BashOperator(
        task_id="ingest_meta",
        bash_command="python /opt/airflow/scripts/ingest_metadata.py",
    )

    ingest_rev = BashOperator(
        task_id="ingest_rev",
        bash_command="python /opt/airflow/scripts/ingest_reviews.py",
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="python /opt/airflow/scripts/preprocess.py",
    )

    ingest_meta >> ingest_rev >> preprocess
