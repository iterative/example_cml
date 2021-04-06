from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Generate some data
X,y = make_classification(1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

# Fit a model
depth = 2
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train,y_train)

# Assess accuracy on held-out data and print the accuracy
acc = clf.score(X_test, y_test)
print(acc)
with open("metrics.txt", 'w') as outfile:
        outfile.write("Accuracy: " + str(acc) + "\n")

# Plot it
disp = plot_confusion_matrix(clf, X_test, y_test, normalize='true',cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')
