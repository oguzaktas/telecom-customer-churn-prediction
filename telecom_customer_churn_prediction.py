
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv("telco_customer_churn.csv")
data.head()

data.columns.values

data.describe()

# Converting Total Charges to a numerical data type
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors="coerce")
data.isnull().sum()


# There are eleven missing values in TotalCharges for some of the customers with zero tenure. We can impute these values with zero as these customers probably haven't paied any bills yet.

# Impute missing values with 0
data["TotalCharges"] = data["TotalCharges"].replace(" ", 0).astype("float32")
# data["TotalCharges"] = data["TotalCharges"].replace(" ", np.nan)

data.dtypes


ax = sns.distplot(data["tenure"], hist=True, kde=False, hist_kws={"edgecolor":"black"}, kde_kws={"linewidth": 6})
ax.set_ylabel("Customers")
ax.set_xlabel("Tenure (months)")
ax.set_title("Number of customers by their tenure")

# Target feature counts
data["Churn"].value_counts(sort=False)

# Creating categorical columns for tenure feature
def tenure_lab(data):
    if data["tenure"] <= 12:
        return "Tenure_0-12"
    elif (data["tenure"] > 12) & (data["tenure"] <= 24):
        return "Tenure_12-24"
    elif (data["tenure"] > 24) & (data["tenure"] <= 48):
        return "Tenure_24-48"
    elif (data["tenure"] > 48) & (data["tenure"] <= 60):
        return "Tenure_48-60"
    elif data["tenure"] > 60:
        return "Tenure_gt_60"
data["tenure_group"] = data.apply(lambda data:tenure_lab(data), axis=1)

# Customer attrition in tenure groups
sns.catplot(x="tenure_group", hue="Churn", kind="count", data=data, aspect=1.6, height=6)

# Removing missing values
data.dropna(inplace=True)

# Replace binary values to numeric values
data["SeniorCitizen"] = data["SeniorCitizen"].replace({1:"Yes", 0:"No"})

# Converting the predictor variable to binary numeric variable
df = data.iloc[:,1:]
df["Churn"].replace(to_replace="Yes", value=1, inplace=True)
df["Churn"].replace(to_replace="No", value=0, inplace=True)
df_dummies = pd.get_dummies(df)
df_dummies.head()

# Correlation of Churn with other features
plt.figure(figsize=(15,8))
df_dummies.corr()["Churn"].sort_values(ascending=False).plot(kind="bar")

# Contracts information
ax = data["Contract"].value_counts().plot(kind="bar", rot=0, width=0.5)
ax.set_ylabel("Number of Customers")
ax.set_title("Number of customers by contract type")

# MonthlyCharges and TotalCharges information
data[["MonthlyCharges", "TotalCharges"]].plot.scatter(x="MonthlyCharges", y="TotalCharges")

# Churn rates of customers
df = data["Churn"].value_counts() * 100.0 / len(data)
ax = df.plot(kind="bar", stacked=True, rot=0, figsize=(9,7))
ax.set_ylabel("% Customers", fontsize=13)
ax.set_xlabel("Churn", fontsize=13)
ax.set_title("Churn Rate", fontsize=13)

# Churn by tenure
sns.boxplot(x=data.Churn, y=data.tenure)

# Churn by contract type
df = data.groupby(["Contract", "Churn"]).size().unstack()
df1 = df.T * 100.0 / df.T.sum()
ax = df1.T.plot(kind="bar", stacked=True, rot=0, figsize=(10,6))
ax.legend(loc="best", prop={"size":13}, title="Churn")
ax.set_title("Churn by contract type", size=13)

# Churn by seniority
df = data.groupby(["SeniorCitizen", "Churn"]).size().unstack()
df1 = df.T * 100.0 / df.T.sum()
ax = df1.T.plot(kind="bar", stacked=True, rot=0, figsize=(9,6))
ax.legend(loc="best", prop={"size":13}, title="Churn")
ax.set_title("Churn by seniority level", size=13)


# Feature Scaling
# Scaling all variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler

y = df_dummies["Churn"].values
X = df_dummies.drop(columns = ["Churn"])
features = X.columns.values
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features

# Model Building (Logistic Regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Splitting the data into training and test sets. The models will be trained on the training set and tested on the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)
model1 = LogisticRegression()
result = model1.fit(X_train, y_train)

# Metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, scorer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import cohen_kappa_score

# Predictions
prediction_test = model1.predict(X_test)
print("Classification report: \n", classification_report(y_test, prediction_test))
print("Accuracy score: {:.5f}\n".format(accuracy_score(y_test, prediction_test)))
print("f1_score: {:.5f}\n".format(f1_score(y_test, prediction_test)))
print("Cohen's kappa score: {:.5f}\n".format(cohen_kappa_score(y_test, prediction_test)))
print("Confusion matrix: \n", confusion_matrix(y_test, prediction_test))


import plotly.offline as py
import plotly.figure_factory as ff

predictions = model1.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
recallscore = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
f1score = f1_score(y_test, predictions) 
kappa_metric = cohen_kappa_score(y_test, predictions)

df = pd.DataFrame({"Accuracy_score": [accuracy],
                   "Recall_score": [recallscore],
                   "Precision": [precision],
                   "f1_score": [f1score],
                   "Area_under_curve": [roc_auc],
                   "Kappa_metric": [kappa_metric]})

model_performance = pd.concat([df], axis = 0).reset_index()
model_performance = model_performance.drop(columns = "index", axis=1)
table = ff.create_table(np.round(model_performance, 5))
py.iplot(table)


from sklearn.model_selection import cross_val_score

# 10 Folds Cross Validation
clf_score = cross_val_score(model1, X_train, y_train, cv=10)
print(clf_score)
clf_score.mean()

# Getting the weights of all variables
weights = pd.Series(model1.coef_[0], index=X.columns.values)
print(weights.sort_values(ascending=False)[:10].plot(kind="bar"))


from keras.models import Sequential
from keras import layers
from keras.layers.core import Dropout

# Artificial neural networks model building with Keras and TensorFlow
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=110)
model2 = Sequential()
input_shape = X_train.shape[1]
model2.add(layers.Dense(1024, input_shape=(input_shape,), activation="relu"))

# Dropout for avoiding overfitting
model2.add(Dropout(0.2))
model2.add(layers.Dense(1024, activation="relu"))
model2.add(Dropout(0.2))
model2.add(layers.Dense(1, activation="sigmoid"))
model2.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model2.summary()

fit_keras = model2.fit(X_train, y_train, epochs=100, verbose=True, 
                       validation_data=(X_test, y_test), batch_size=30)

accuracy = model2.evaluate(X_train, y_train, verbose=False)
print("Training score: {:.5f}".format(accuracy[0]))
print("Training accuracy: {:.5f}\n".format(accuracy[1]))

accuracy = model2.evaluate(X_test, y_test, verbose=False)
print("Testing score: {:.5f}".format(accuracy[0]))
print("Testing accuracy: {:.5f}\n".format(accuracy[1]))


# Model building and performance of model (K-Nearest Neighbors Classifier)
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=120)
model3 = KNeighborsClassifier()
result = model3.fit(X_train, y_train)

predictions = model3.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
recallscore = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
f1score = f1_score(y_test, predictions) 
kappa_metric = cohen_kappa_score(y_test, predictions)

df = pd.DataFrame({"Accuracy_score": [accuracy],
                   "Recall_score": [recallscore],
                   "Precision": [precision],
                   "f1_score": [f1score],
                   "Area_under_curve": [roc_auc],
                   "Kappa_metric": [kappa_metric]})

model_performance = pd.concat([df], axis = 0).reset_index()
model_performance = model_performance.drop(columns = "index", axis=1)
table = ff.create_table(np.round(model_performance, 5))
py.iplot(table)


# Model building and performance of model (Decision Tree)
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=120)
model4 = DecisionTreeClassifier()
result = model4.fit(X_train, y_train)

predictions = model4.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
recallscore = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
f1score = f1_score(y_test, predictions) 
kappa_metric = cohen_kappa_score(y_test, predictions)

df = pd.DataFrame({"Accuracy_score": [accuracy],
                   "Recall_score": [recallscore],
                   "Precision": [precision],
                   "f1_score": [f1score],
                   "Area_under_curve": [roc_auc],
                   "Kappa_metric": [kappa_metric]})

model_performance = pd.concat([df], axis = 0).reset_index()
model_performance = model_performance.drop(columns = "index", axis=1)
table = ff.create_table(np.round(model_performance, 5))
py.iplot(table)


# Model building and performance of model (Random Forest)
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=120)
model5 = RandomForestClassifier()
result = model5.fit(X_train, y_train)

predictions = model5.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
recallscore = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
f1score = f1_score(y_test, predictions) 
kappa_metric = cohen_kappa_score(y_test, predictions)

df = pd.DataFrame({"Accuracy_score": [accuracy],
                   "Recall_score": [recallscore],
                   "Precision": [precision],
                   "f1_score": [f1score],
                   "Area_under_curve": [roc_auc],
                   "Kappa_metric": [kappa_metric]})

model_performance = pd.concat([df], axis = 0).reset_index()
model_performance = model_performance.drop(columns = "index", axis=1)
table = ff.create_table(np.round(model_performance, 5))
py.iplot(table)


# Model building and performance of model (Gaussian Naive Bayes)
from sklearn.naive_bayes import GaussianNB

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=110)
model6 = RandomForestClassifier()
result = model6.fit(X_train, y_train)

predictions = model6.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
recallscore = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
f1score = f1_score(y_test, predictions) 
kappa_metric = cohen_kappa_score(y_test, predictions)

df = pd.DataFrame({"Accuracy_score": [accuracy],
                   "Recall_score": [recallscore],
                   "Precision": [precision],
                   "f1_score": [f1score],
                   "Area_under_curve": [roc_auc],
                   "Kappa_metric": [kappa_metric]})

model_performance = pd.concat([df], axis = 0).reset_index()
model_performance = model_performance.drop(columns = "index", axis=1)
table = ff.create_table(np.round(model_performance, 5))
py.iplot(table)


# Model building and performance of model (Support Vector Machine)
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=120)
model7 = RandomForestClassifier()
result = model7.fit(X_train, y_train)

predictions = model7.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
recallscore = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
f1score = f1_score(y_test, predictions) 
kappa_metric = cohen_kappa_score(y_test, predictions)

df = pd.DataFrame({"Accuracy_score": [accuracy],
                   "Recall_score": [recallscore],
                   "Precision": [precision],
                   "f1_score": [f1score],
                   "Area_under_curve": [roc_auc],
                   "Kappa_metric": [kappa_metric]})

model_performance = pd.concat([df], axis = 0).reset_index()
model_performance = model_performance.drop(columns = "index", axis=1)
table = ff.create_table(np.round(model_performance, 5))
py.iplot(table)


# Model building and performance of model (MLP Neural Network Classifier)
from sklearn.neural_network import MLPClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
model8 = RandomForestClassifier()
result = model8.fit(X_train, y_train)

predictions = model8.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
recallscore = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
f1score = f1_score(y_test, predictions) 
kappa_metric = cohen_kappa_score(y_test, predictions)

df = pd.DataFrame({"Accuracy_score": [accuracy],
                   "Recall_score": [recallscore],
                   "Precision": [precision],
                   "f1_score": [f1score],
                   "Area_under_curve": [roc_auc],
                   "Kappa_metric": [kappa_metric]})

model_performance = pd.concat([df], axis = 0).reset_index()
model_performance = model_performance.drop(columns = "index", axis=1)
table = ff.create_table(np.round(model_performance, 5))
py.iplot(table)


# Model building and performance of model (Ensemble Gradient Boosting Classifier)
from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
model9 = GradientBoostingClassifier()
result = model9.fit(X_train, y_train)

predictions = model9.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
recallscore = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
f1score = f1_score(y_test, predictions) 
kappa_metric = cohen_kappa_score(y_test, predictions)

df = pd.DataFrame({"Accuracy_score": [accuracy],
                   "Recall_score": [recallscore],
                   "Precision": [precision],
                   "f1_score": [f1score],
                   "Area_under_curve": [roc_auc],
                   "Kappa_metric": [kappa_metric]})

model_performance = pd.concat([df], axis = 0).reset_index()
model_performance = model_performance.drop(columns = "index", axis=1)
table = ff.create_table(np.round(model_performance, 5))
py.iplot(table)


# Model building and performance of model (AdaBoost Classifier)
from sklearn.ensemble import AdaBoostClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
model10 = AdaBoostClassifier()
result = model10.fit(X_train, y_train)

predictions = model10.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
recallscore = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
f1score = f1_score(y_test, predictions) 
kappa_metric = cohen_kappa_score(y_test, predictions)

df = pd.DataFrame({"Accuracy_score": [accuracy],
                   "Recall_score": [recallscore],
                   "Precision": [precision],
                   "f1_score": [f1score],
                   "Area_under_curve": [roc_auc],
                   "Kappa_metric": [kappa_metric]})

model_performance = pd.concat([df], axis = 0).reset_index()
model_performance = model_performance.drop(columns = "index", axis=1)
table = ff.create_table(np.round(model_performance, 5))
py.iplot(table)


# Model building and performance of model (XGBoost Classifier)
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
model11 = AdaBoostClassifier()
result = model11.fit(X_train, y_train)

predictions = model11.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
recallscore = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
f1score = f1_score(y_test, predictions) 
kappa_metric = cohen_kappa_score(y_test, predictions)

df = pd.DataFrame({"Accuracy_score": [accuracy],
                   "Recall_score": [recallscore],
                   "Precision": [precision],
                   "f1_score": [f1score],
                   "Area_under_curve": [roc_auc],
                   "Kappa_metric": [kappa_metric]})

model_performance = pd.concat([df], axis = 0).reset_index()
model_performance = model_performance.drop(columns = "index", axis=1)
table = ff.create_table(np.round(model_performance, 5))
py.iplot(table)


# Model building and performance of model (LGBM Classifier)
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
model11 = AdaBoostClassifier()
result = model11.fit(X_train, y_train)

predictions = model11.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
recallscore = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
f1score = f1_score(y_test, predictions) 
kappa_metric = cohen_kappa_score(y_test, predictions)

df = pd.DataFrame({"Accuracy_score": [accuracy],
                   "Recall_score": [recallscore],
                   "Precision": [precision],
                   "f1_score": [f1score],
                   "Area_under_curve": [roc_auc],
                   "Kappa_metric": [kappa_metric]})

model_performance = pd.concat([df], axis = 0).reset_index()
model_performance = model_performance.drop(columns = "index", axis=1)
table = ff.create_table(np.round(model_performance, 5))
py.iplot(table)

