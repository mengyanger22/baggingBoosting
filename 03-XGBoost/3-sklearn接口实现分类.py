from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import xgboost as xgb
from xgboost import plot_importance, plot_tree

import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target
feature_name = iris.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

model = xgb.XGBClassifier(max_depth=5, n_estimators=50, silent=True, objective="multi:softmax", feature_names=feature_name)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# print("accuracy:", accuracy)
# plot_importance(model)
# plt.show()
# plot_tree(model, num_trees=5)
# plt.show()