import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Read in data
df = pd.read_csv("data/winequality-red-test.csv")
df

df['quality'].value_counts()

df.info()

df.columns

df.drop(["Id"],axis = 1,inplace = True)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from sklearn.metrics import accuracy_score

X = df.drop( "quality",axis=1)
y = df["quality"]

# use 70% of the data for training and 30% for testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=50)

# RandomForest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50,max_depth=10, random_state=101,class_weight='balanced')

# Fit a model
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

with open("metrics.txt", 'w') as outfile:
        outfile.write("F1 Score :" + str(round(f1_score(y_pred,y_test,average = "weighted"), 2)) + "\n")
        outfile.write("Report:\n" + classification_report(y_test, y_pred) + "\n")


# Plot it
disp = plot_confusion_matrix(rfc, X_test, y_test, normalize='true',cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')

# save model file
with open("model/model.pkcl", 'wb') as pickle_file:
    pickle.dump(clf, pickle_file)
