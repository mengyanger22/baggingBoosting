from sklearn.datasets import fetch_california_housing
import xgboost as xgb
from xgboost import plot_importance, plot_tree

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

california = fetch_california_housing()
X, y = california.data, california.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=50, silent=True, objective="reg:gamma")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# plot_importance(model)
# plt.show()

plot_tree(model, num_trees=17)
plt.show()