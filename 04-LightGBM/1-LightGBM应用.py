import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


breast = load_breast_cancer()
X, y = breast.data, breast.target
feature_name = breast.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

boost_round = 50
early_stop_rounds = 10
params = {
    "boosting_type" : "gbdt",
    "objective" : "regression",
    "metric" : {"12", "auc"},
    "num_leaves" : 31,
    "learning_rate" : 0.05,
    "feature_fraction" : 0.9,
    "bagging_fraction" : 0.8,
    "bagging_freq" : 5,
    "verbose" : 1,
}

results = {}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=boost_round,
                valid_sets=(lgb_eval, lgb_train),
                valid_names=("validate", "train"),
                early_stopping_rounds=early_stop_rounds,
                evals_result=results
                )

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# print(y_pred)

# lgb.plot_metric(results)
# plt.show()

lgb.plot_importance(gbm, importance_type="split")
plt.show()