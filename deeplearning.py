import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# load data
data = pd.read_csv('dataset_deeplearning.csv')
data.dropna(inplace=True)

# delect data without invited data and transaction data
data['delect_label'] = data['parent_number'] + data['child_number']
data = data[data['delect_label'] > 0]
data['delect_label'] = data['avg_value']
data = data[data['delect_label'] > 0]


label = data['label']
data = data.drop(['address', 'label', 'delect_label'], axis=1)
X = data
y = label
# X.to_csv('data/X.csv', index=False)
# y.to_csv('data/y.csv', index=False)
X = X.values
y = y.values

# random forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_split=15, min_samples_leaf=2)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

