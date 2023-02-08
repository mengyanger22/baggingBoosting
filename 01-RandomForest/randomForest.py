import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns

# 中文乱码
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

def match(s1, s2):
    l1, l2 = [], []
    if "_" in s1:
        l1 = s1.split("_")
    if "_" in s2:
        l2 = s2.split("_")
    if not l1:
        n = len(s1)
        l, r = 0, 0
        while r < n:
            while r < n and s1[r].islower():
                r += 1
            l1.append(s1[l].lower() + s1[l+1:r])
            l = r
            r += 1
    if not l2:
        n = len(s2)
        l, r = 0, 0
        while r < n:
            while r < n and s2[r].islower():
                r += 1
            l2.append(s2[l].lower() + s2[l+1:r])
            l = r
            r += 1
    
    # print(l1, l2)
    ans = 0
    for i in l1:
        for j in l2:
            if i == j:
                ans += 1
    
    return ans

path = "D:\linux\opt\pjs\\baggingBoosting\data\HR_comma_sep.csv"
df = pd.read_csv(path, index_col=None)
# print(df.isnull().any())
# print(df.head())
old_to_new = {"sales":"department", "left":"turnover"}
old_cols = df.columns.values.tolist()
# print(old_cols)
new_cols = ["satisfaction", "evaluation", "projectCount", "averageMonthlyHours", "yearsAtCompany", "workAccident", "promotion"]
for col in old_cols:
    if col not in old_to_new:
        tmp, new_col = 0, ""
        for i in new_cols:
            t1 = match(col, i)
            if t1 > tmp:
                t1 = tmp
                new_col = i
        if new_col:
            old_to_new[col] = new_col

# print(old_to_new)

df = df.rename(columns=old_to_new)
front = df["turnover"]
df.drop(labels=["turnover"], axis=1, inplace=True)
df.insert(0, "turnover", front)
# print(df.head())
# print(df.shape)
# print(df.dtypes)
turnover_rate = df.turnover.value_counts() / len(df)
# print(turnover_rate)
# print(df.describe())
turnover_Summary = df.groupby("turnover")
# print(turnover_Summary.mean())
corr = df.corr()
# sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
# print(corr)
# plt.show()

emp_population = df["satisfaction"][df["turnover"] == 0].mean()
emp_turnover_satisfaction = df[df["turnover"] == 1]["satisfaction"].mean()
# print("未离职满意度：" + str(emp_population))
# print("离职满意度：" + str(emp_turnover_satisfaction))

import scipy.stats as stats
ttest_res = stats.ttest_1samp(a=df[df["turnover"] == 1]["satisfaction"], popmean=emp_population)
# print(ttest_res)
degree_freedom = len(df[df["turnover"]==1])
# 临界值
LQ = stats.t.ppf(0.025, degree_freedom)
RQ = stats.t.ppf(0.975, degree_freedom)
# print("t分布 左边界：" + str(LQ))
# print("t分布 有边界：" + str(RQ))

# fig = plt.figure(figsize=(15, 4),)
# ax = sns.kdeplot(df.loc[(df["turnover"] == 0), "evaluation"], color="b", fill=True, label="no turnover")
# ax = sns.kdeplot(df.loc[(df["turnover"] == 1), "evaluation"], color="r", fill=True, label="turnover")
# ax.set(xlabel="工作评价", ylabel="频率")
# plt.title("工作评价的概率密度函数 - 离职 VS 未离职")
# plt.show()

# fig = plt.figure(figsize=(15, 4))
# ax = sns.kdeplot(df.loc[(df["turnover"] == 0), "averageMonthlyHours"], color="b", fill=True, label="no turnover")
# ax = sns.kdeplot(df.loc[(df["turnover"] == 1), "averageMonthlyHours"], color="r", fill=True, label="turnover")
# ax.set(xlabel="月工作时长", ylabel="频率")
# plt.title("月工作时长  离职 VS 未离职")
# plt.show()

# fig = plt.figure(figsize=(15, 4))
# ax = sns.kdeplot(df.loc[(df["turnover"] == 0), "satisfaction"], color="b", fill=True, label="no turnover")
# ax = sns.kdeplot(df.loc[(df["turnover"] == 1), "satisfaction"], color="r", fill=True, label="turnover")
# plt.title("员工满意度  离职 VS 未离职")
# plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, roc_auc_score

df["department"] = df["department"].astype("category").cat.codes
df["salary"] = df["salary"].astype("category").cat.codes

target_name = "turnover"
X = df.drop(target_name, axis=1)
y = df[target_name]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123, stratify=y)
# print(df.head())

# 决策树和随机森林Desision Tree  VS  Random Forest
import pydotplus
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus


dtree = tree.DecisionTreeClassifier(
    criterion="entropy",
    # max_depth=3,
    min_weight_fraction_leaf=0.01,
)

dtree = dtree.fit(X_train, y_train)
dt_roc_auc = roc_auc_score(y_test, dtree.predict(X_test))
# print("决策树 AUC = %2.2f" % dt_roc_auc)
# print(classification_report(y_test, dtree.predict(X_test)))

feature_names = df.columns[1:]
# dot_data = StringIO()
# export_graphviz(dtree, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True, feature_names=feature_names, class_names=["0", "1"]
#                 )
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png("diabetes.png")
# Image(graph.create_png())

importances = dtree.feature_importances_
feat_names = df.drop(["turnover"], axis=1).columns
indices = np.argsort(importances)[::-1]

# plt.figure(figsize=(12, 6))
# plt.title("Feature importance by Descision Tree")
# plt.bar(range(len(indices)), importances[indices], color="lightblue", align="center")
# plt.step(range(len(indices)), np.cumsum(importances[indices]), where="mid", label="Cumulative")
# plt.xticks(range(len(indices)), feat_names[indices], rotation="vertical", fontsize=14)
# plt.xlim([-1, len(indices)])
# plt.show()



rf = RandomForestClassifier(
    criterion="entropy",
    n_estimators=3,
    max_depth=None,
    min_samples_split=10,
    # min_weight_fraction_leaf=0.02
)

rf.fit(X_train, y_train)
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
# print("随机森林 AUC = %2.2f" % rf_roc_auc)
# print(classification_report(y_test, rf.predict(X_test)))

# Estimators = rf.estimators_
# for index, model in enumerate(Estimators):
#     dot_data = StringIO()
#     export_graphviz(model, out_file=dot_data,
#                     feature_names=df.columns[1:],
#                     class_names=["0", "1"],
#                     filled=True, rounded=True,
#                     special_characters=True
#                     )
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#     graph.write_png("Rf{}.png".format(index))
#     plt.figure(figsize=(20, 20))
#     plt.imshow(plt.imread("Rf{}.png".format(index)))
#     plt.axis("off")

importances = rf.feature_importances_
feat_names = df.drop(["turnover"], axis=1).columns
indices = np.argsort(importances)[::-1]
# plt.figure(figsize=(12, 6))
# plt.title("Featrue importances by RandomForest")
# plt.bar(range(len(indices)), importances[indices], color="lightblue", align="center")
# plt.step(range(len(indices)), np.cumsum(importances[indices]), where="mid", label="Cumulative")
# plt.xticks(range(len(indices)), feat_names[indices], rotation="vertical", fontsize=14)
# plt.xlim([-1, len(indices)])
# plt.show()

from sklearn.metrics import roc_curve
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dtree.predict_proba(X_test)[:, 1])
plt.figure()

plt.plot(rf_fpr, rf_tpr, label="Random Forest (area = %0.2f)" % rf_roc_auc)
plt.plot(dt_fpr, dt_tpr, label="Decision Tree (area = %0.2f)" % dt_roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Graph")
plt.legend(loc="lower right")
plt.show()