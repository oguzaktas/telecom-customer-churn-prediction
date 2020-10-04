import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, scorer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import cohen_kappa_score
import plotly.offline as py
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler

def build_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=120)
    model = SVC()
    result = model.fit(X_train, y_train)

    return result

def model_performance(model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=120)
    result = model.fit(X_train, y_train)

    predictions = model.predict(X_test)
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

if __name__ == '__main__':
    model = build_model()
    model_performance(model)
