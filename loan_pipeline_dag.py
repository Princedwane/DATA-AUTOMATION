from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

# Adding project path so Airflow can find the pipeline functions
sys.path.insert(0, '/home/princedwane/Desktop/DATA_AUTOMATION')

# Importing pipeline functions
from pipeline_v2 import (
    ingest_data,
    validate_data,
    transform_data,
    load_data_output,
    generate_report,
    generate_eda_report
)

# Default arguments for all tasks
default_args = {
    'owner': 'Princedwane John Martine',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# Defining the DAG
with DAG(
    dag_id='loan_pipeline',
    description='Automated loan data pipeline',
    default_args=default_args,
    start_date=datetime(2026, 7, 4),
    schedule_interval='@daily',
    catchup=False,
) as dag:

    # Task 1 — Ingest
    def run_ingest():
        return ingest_data('/home/princedwane/Desktop/DATA_AUTOMATION/loan_approval_dataset.csv')

    # Task 2 — Validate
    def run_validate():
        df = ingest_data('/home/princedwane/Desktop/DATA_AUTOMATION/loan_approval_dataset.csv')
        return validate_data(df)

    # Task 3 — Transform
    def run_transform():
        df = ingest_data('/home/princedwane/Desktop/DATA_AUTOMATION/loan_approval_dataset.csv')
        df = validate_data(df)
        transform_data(df)

    # Task 4 — Load and Report
    def run_load_report():
        df = ingest_data('/home/princedwane/Desktop/DATA_AUTOMATION/loan_approval_dataset.csv')
        df = validate_data(df)
        df = transform_data(df)
        load_data_output(df)
        generate_report(df)
        generate_eda_report(df)

    ingest_task = PythonOperator(task_id='ingest', python_callable=run_ingest)
    validate_task = PythonOperator(task_id='validate', python_callable=run_validate)
    transform_task = PythonOperator(task_id='transform', python_callable=run_transform)
    load_report_task = PythonOperator(task_id='load_and_report', python_callable=run_load_report)

    # Task order — defines the pipeline flow
    ingest_task >> validate_task >> transform_task >> load_report_task