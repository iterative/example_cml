### Data Pipeline desarrollado en Airflow

from __future__ import print_function
#from builtins import range
#from pprint import pprint
from airflow.utils.dates import days_ago
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from datetime import datetime, date

#import sys
#sys.path.append('/home')
#from aimodels.universal_function import *


args = {
    'owner': 'Test',
    'start_date': datetime(2022, 1, 31)
}

dag = DAG(
    dag_id='DataPipeline',
    default_args=args,
    schedule_interval='@monthly',
    tags=['datapipeline']
)

def print_context(ds, **kwargs):
    #pprint(kwargs)
    print(ds)
    return 'Whatever you return gets printed in the logs'

run_this = PythonOperator(
    task_id='print_the_context',
    provide_context=True,
    python_callable=print_context,
    dag=dag,
)

### 7 Tareas que inician con la lectura de los datos, construcción de las features de entrenamiento y generación de la tabla con los features de entrenamiento.
### El Pipeline tiene un task para generar cada feature y estas se ejecutan en paralelo, excepto la tarea de lectura y la tarea de escritura que se ejecuta justo cuando estén creadas todas la features.
### El archivo final se guarda en formato parquet desde spark para optimizar siguientes operaciones requeridas sobre él y que puedan ser desarrolladas con spark.

def Read_data(ti):
    
    df = pd.read_csv('/opt/conda/envs/airflow/lib/python3.8/site-packages/airflow/dags/Test/dataset_credit_risk.csv')
    df = df.sort_values(by=["id", "loan_date"])
    df = df.reset_index(drop=True)
    df["loan_date"] = pd.to_datetime(df.loan_date)
    ti.xcom_push(key='df', value=df)
    
    
def Get_nb_previous_loans(ti):
    
    df=ti.xcom_pull(key='df')
    df_grouped = df.groupby("id")
    df["nb_previous_loans"] = df_grouped["loan_date"].rank(method="first") - 1
    ti.xcom_push(key='df', value=df)
    
    
def Get_avg_amount_loans_previous(ti):
    
    df=ti.xcom_pull(key='df')
    avg_amount_loans_previous = pd.Series()
    for user in df.id.unique():
        df_user = df.loc[df.id == user, :]
        avg_amount_loans_previous = avg_amount_loans_previous.append(df_user["loan_amount"].rolling(df_user.shape[0], min_periods=1).mean().shift(periods=1))
    df["avg_amount_loans_previous"] = avg_amount_loans_previous
    ti.xcom_push(key='df', value=df)
    

def Get_age(ti):
    
    df=ti.xcom_pull(key='df')
    df['birthday'] = pd.to_datetime(df['birthday'], errors='coerce')
    df['age'] = (pd.to_datetime('today').normalize() - df['birthday']).dt.days // 365
    ti.xcom_push(key='df', value=df)
    

def Get_years_on_the_job(ti):
    
    df=ti.xcom_pull(key='df')
    df['job_start_date'] = pd.to_datetime(df['job_start_date'], errors='coerce')
    df['years_on_the_job'] = (pd.to_datetime('today').normalize() - df['job_start_date']).dt.days // 365
    ti.xcom_push(key='df', value=df)


def Get_flag_own_car(ti):
    
    df=ti.xcom_pull(key='df')
    df['flag_own_car'] = df.flag_own_car.apply(lambda x : 0 if x == 'N' else 1)
    ti.xcom_push(key='df', value=df)


def Write_data(ti):
    
    df=ti.xcom_pull(key='df')
    df = df[['id', 'age', 'years_on_the_job', 'nb_previous_loans', 'avg_amount_loans_previous', 'flag_own_car', 'status']]
    
    ### Esquema de la tabla de entrenamiento en spark
    
    schema = StructType([
    StructField("id", IntegerType()),
    StructField("age", IntegerType()),
    StructField("years_on_the_job", FloatType()),
    StructField("nb_previous_loans", FloatType()),
    StructField("avg_amount_loans_previous", FloatType()),
    StructField("flag_own_car", StringType()),
    StructField("status", IntegerType())    
    ])
    
    ### Parh donde se guardará el archivo
    
    path = "wasbs://data@stg0ia0prod0001.blob.core.windows.net/data_train_spark/"
    spark = spark_init(name = "Test",front=False)
    sdf = spark.createDataFrame(df,schema)
    sdf.write.format("parquet").mode("overwrite").save(path)

### Creación de las tareas
    
task_1 = PythonOperator(
    task_id='Read_data',
    python_callable=Read_data,
    op_kwargs={},
    dag=dag,
)

task_2 = PythonOperator(
    task_id='Get_nb_previous_loans',
    python_callable=Get_nb_previous_loans,
    op_kwargs={},
    dag=dag,
)

task_3 = PythonOperator(
    task_id='Get_avg_amount_loans_previous',
    python_callable=Get_avg_amount_loans_previous,
    op_kwargs={},
    dag=dag,
)

task_4 = PythonOperator(
    task_id='Get_age',
    python_callable=Get_age,
    op_kwargs={},
    dag=dag,
)

task_5 = PythonOperator(
    task_id='Get_years_on_the_job',
    python_callable=Get_years_on_the_job,
    op_kwargs={},
    dag=dag,
)

task_6 = PythonOperator(
    task_id='Get_flag_own_car',
    python_callable=Get_flag_own_car,
    op_kwargs={},
    dag=dag,
)

task_7 = PythonOperator(
    task_id='Write_data',
    python_callable=Write_data,
    op_kwargs={},
    dag=dag,
)

### Scheduler de las tareas y envío a ejecución

run_this >> task_1 >> [task_2, task_3, task_4, task_5, task_6] >> task_7
