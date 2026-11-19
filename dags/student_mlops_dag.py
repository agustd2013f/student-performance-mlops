from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "maria_fernanda",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="student_risk_mlops_pipeline",
    default_args=default_args,
    description="Orquestacion end-to-end del pipeline MLOps de riesgo estudiantil",
    schedule="@weekly",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    tags=["mlops", "student-risk"],
) as dag:

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command="cd /opt/airflow/project && /home/airflow/ml_venv/bin/python src/prepare_data.py",
    )

    train_and_register = BashOperator(
        task_id="train_mlflow_champion",
        bash_command="mkdir -p /tmp/mlflow_run && ln -sfn /opt/airflow/project/src /opt/airflow/project/data /opt/airflow/project/models /opt/airflow/project/reports /tmp/mlflow_run/ && cd /tmp/mlflow_run && /home/airflow/ml_venv/bin/python src/train_mlflow.py",
    )

    monitor_drift = BashOperator(
        task_id="monitor_evidently_drift",
        bash_command="cd /opt/airflow/project && /home/airflow/ml_venv/bin/python src/generate_report.py && /home/airflow/ml_venv/bin/python src/performance_monitor.py",
    )

    prepare_data >> train_and_register >> monitor_drift