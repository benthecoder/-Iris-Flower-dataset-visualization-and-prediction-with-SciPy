# Load libraries
from typing import List, Any, Union, Tuple
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load the data_set
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data_set = pandas.read_csv(url, names=names)

# SUMMARIZE THE DATASET

# Dimensions of data_set

# shape
print(data_set.shape)  # (rows, attributes)

# head
print(data_set.head(20))  # first 20 rows of data

# STATISTICAL SUMMARY

# descriptions
print(data_set.describe())  # count, mean, min and max etc.

# Class Distribution
print(data_set.groupby('class').size())  # no. of instances(rows) that belong to each class

# DATA VISUALIZATION

# Univariate Plots - to better understand each attributes

# box and whisker plots - clearer idea on the input attributes
data_set.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

# histograms - get an idea of the distribution
data_set.hist()
plt.show()

# MULTIVARIATE PLOTS

# scatter plot matrix - helpful to spot structured relationships btn input variables
scatter_matrix(data_set)
plt.show()

# EVALUATE SOME ALGORITHM

# Create a validation dataset - Operating on unseen data to estimate accuracy

# Split-out validation dataset
array = data_set.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20  # split dataset into 80(train models) and 20(validation)
seed: int = 7  # generates a random number
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)
# TEST HARNESS

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'  # a metric to evaluate models (%)

# BUILD MODELS

# Logistic Regression (LR)
# Linear Discriminant Analysis (LDA)

# K-Nearest Neighbors (KNN)
# Classification and Regression Trees (CART)
# Gaussian Naive Bayes (NB)
# Support Vector Machines (SVM)

# Spot Check Algorithms
models: List[Union[Tuple[str, LogisticRegression], Tuple[str, LinearDiscriminantAnalysis],
                   Tuple[str, KNeighborsClassifier], Tuple[str, DecisionTreeClassifier], Tuple[str, GaussianNB],
                   Tuple[str, SVC]]] = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
                                        ('LDA', LinearDiscriminantAnalysis()), ('KNN', KNeighborsClassifier()),
                                        ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()),
                                        ('SVM', SVC(gamma='auto'))]
# evaluate each model in turn
results: List[Any] = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results: object = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg: str = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# SVM has largest estimated accuracy score

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# MAKE PREDICTIONS

# independent final check on accuracy on best model
# a validation set is important in case of mishaps during training(over fitting/data leak)

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# confusion matrix provides indication of errors
