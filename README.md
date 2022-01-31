A continuación se detallan los pasos que se desarrollaron para resolver el desafío en su totalidad. Dentro de cada notebook y archivo de python se documenta más en detalle.

1. En MLE_challenge_Features_engineering_Notebook_1.ipynb se optimizó el cálculo de la feature "avg_amount_loans_previous" obteniendo la feature para todo el set de entrenamiento en --- 152.88485550880432 seconds --- unas 6 veces más rápido que el método tradicional que demora --- 820.6095194816589 seconds ---.

2. En datapipeline.py se construye un Data Pipeline de Spark utilizando Airflow donde se generan las features de entrenamiento en una task diferente para cada una de ellas, estas task corren en paralelo como se muestra en DataPipeline.JPG y DataPipelineDagTask.JPG excepto la task que lee los datos originales y la task que gener el archivo de entrenamiento en formaro parquet como se observa en DataTrainSpark.JPG. Esta últina se ejecuta luego de tener todas las features generadas.

3. En ApiFeatures.ipynb se construye una API utilizando un micro servivo de flask y que dado un id, retorna las features de entrenamiento.

4. En ApiPredictions.ipynb se se construye de nuevo otra API utilizando un micro servivo de flask que a su vez invoca la primera API para obtener de ellas las features de entrenamiento dado un id, y con estos datos hacemos una predicción desde esta API.

5. Se integró github con CML (Continuous Machine Learning), herramienta open-source utilizada para continuous integration & delivery (CI/CD) como parte del flujo de trabajo de MLOps. En el repositorio está oculto el archivo /.github/workflows/cnl.yaml que no es otra cosa que el script donde escribes comando cml para enriquecer daca commit y push, estas se visualizan en los Actions de github, como se ve en MlOps1.JPG, MlOps3.JPG y MlOps3.JPG. En requirements.txt están las bibliotecas que se requieren para completar la integración.

6. Cada vez que datapipeline.py se modifique y se lleva al repositorio via push se ejecuta el script de CML que te permite de forma adicional incorporar métricas y plots relacionadas con el Pipeline si hicieran falta incorpotrarlas.

7. Como consideraciones finales, fue un hermoso desafío poder completar este Challenge. El puno 1. a mi parecer fue el más hermoso, se puede aprender muvgo de ese punto y tratar de optimizarlo. Las API por tiempo las construí en flask pero puede usasrse otras herrmaientas y lenguajes. Apache Airflow es una herramienta super poderosa que nos permite construir y paralelizar tareas bin de PySpark o Spark a través de un .jar. Finalmente la Integración y Entrega continua ofrece un sin fín de herramientas que bien valen la pena detallar, les comparto un link de git donde se detallan muchas de ellas: https://github.com/kelvins/awesome-mlops#cicd-for-machine-learning.

Como observación final, trabajé desde una máuina virtual en jupyter-lab en un ambiente que preparé para correr airflow y demás tecnologías. En mi máquina local no hubiera lacanzado terminar en 2 días que lo necesitaba para cumplir con los compromisos de mi trabajo actual. Es por ello que la data que se genera a lo largo del proyecto está en mi máquina virtual.



