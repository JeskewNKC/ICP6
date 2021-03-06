from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.read_csv("iris.csv")
y = np.array(df[['class']])
x = np.array(df.drop(['class'], axis=1))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1) # default for test_size = 0.25

from sklearn.svm import LinearSVC

model = LinearSVC(max_iter= 10000)
model.fit(x_train, y_train.ravel())
y_model = model.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_model))

