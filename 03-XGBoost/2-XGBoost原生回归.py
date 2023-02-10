from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import xgboost as xgb
from xgboost import plot_importance, plot_tree

import matplotlib.pyplot as plt

california = fetch_california_housing()
X, y = california.data, california.target
feature_name = california.feature_names

# print(feature_name)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

params = {
    "booster" : "gbtree",
    "objective" : "reg:gamma",
    "gamma" : 0.1,
    "max_depth" : 5,
    "lambda" : 3,
    "subsample" : 0.7,
    "colsample_bytree" : 0.7,
    "min_child_weight" : 3,
    "silent" : 1,
    "eta" : 0.1,
    "seed" : 1000,
    "nthread" : 4,
}

plst = list(params.items())

dtrain = xgb.DMatrix(X_train, y_train, feature_names=feature_name)
dtest = xgb.DMatrix(X_test, feature_names=feature_name)
num_rounds = 30
model = xgb.train(plst, dtrain, num_rounds)
y_pred = model.predict(dtest)

# plot_importance(model, importance_type="weight")
# plt.show()
# plot_tree(model, num_trees=17)
# plt.show()

saved_model_path = "D:\linux\opt\pjs\\baggingBoosting\saved_models"
model.dump_model(saved_model_path+"\model2.txt")