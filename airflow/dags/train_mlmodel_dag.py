from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.docker_operator import DockerOperator

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2024, 4, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'docker_container_weekly',
    default_args=default_args,
    description='A DAG to start a Docker container every week',
    schedule_interval=timedelta(days=7),
)

docker_task = DockerOperator(
    task_id='run_docker_container',
    image='rakuten_cf:latest',  #TODO: Replace with ML Train Image
    container_name='rakuten_cf',
    docker_url="tcp://docker-socket-proxy:2375",
    api_version='1.37',
    network_mode="bridge",
    # api_version='auto',
    auto_remove=True,
    dag=dag,
)

docker_task

## Run on other server / not tested
# dag = DAG(
#     'docker_container_weekly_on_remote_server',
#     default_args=default_args,
#     description='A DAG to start a Docker container every week on a remote server',
#     schedule_interval=timedelta(days=7),
# )
#
# ssh_command = """
# docker run your_docker_image:latest
# """
#
# docker_task = SSHOperator(
#     task_id='run_docker_container_on_remote_server',
#     ssh_conn_id='your_ssh_connection',  # Specify your SSH connection ID configured in Airflow
#     command=ssh_command,
#     dag=dag,
# )
#
# docker_task
