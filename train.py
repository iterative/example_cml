from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json
import os
import numpy as np

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")

# Fit a model
depth = 5
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = clf.score(X_test, y_test)

os.mkdir("./metrics/")

with open("./metrics/metrics.json", 'w') as outfile:
    json.dump({"accuracy": acc}, outfile)

with open("./metrics/classification_report.json", "w") as outfile:
    json.dump(classification_report(y_test, y_pred, output_dict=True), outfile)

with open("./predictions.csv", "w") as writefile:
    writefile.write("actual,predicted")
    writefile.write("\n")
    for line in zip(y_test, y_pred):
        writefile.write(f"{y_test},{y_pred}")
        writefile.write("\n")

