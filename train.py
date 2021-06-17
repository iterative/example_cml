from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import csv
import dvc.api
import pandas as pd
from sklearn.metrics import precision_score
import mlflow

GITHUB_TOKEN = os.getenv('REPO_TOKEN')
url = 'https://' + GITHUB_TOKEN + ':@' + 'github.com/healiosuk/ML-project-template'
print(url)
MLFLOW_URL = os.getenv('MLFLOW_URL')

with dvc.api.open(
    'data_folder/iris.csv',
    repo=url,
    rev='master'
) as fd:
    data = pd.read_csv(fd)
    # fd is a file descriptor which can be processed normally
print(len(data))
# Read in data
# test
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")


# Fit a model
depth = 5
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train,y_train)

acc = clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
print(acc)
precision = precision_score(y_test, y_pred, average='macro')
print(precision)

mlflow.set_tracking_uri(MLFLOW_URL)
# set experiment name to whatever you want, if it doesn't exist it will be created.
mlflow.set_experiment("test mlflow with cml")
with mlflow.start_run(run_name="run 2") as run:
    
    # add parameters for tuning
    num_estimators = 100
    mlflow.log_param("num_estimators",num_estimators)

    # train the model
    # this will save the model locally or to the S3 bucket if using a server which is what we do
    mlflow.sklearn.log_model(clf, "test-cml-model",  registered_model_name="random-forest-model")


df = pd.DataFrame(data = {"Value":[acc,precision]}, index=["accuracy","precision"])
df.index.names= ['Metric']
with open("metrics.txt", "w") as outfile:
    outfile.write(df.to_markdown())

# Plot it
disp = plot_confusion_matrix(clf, X_test, y_test, normalize='true',cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')

