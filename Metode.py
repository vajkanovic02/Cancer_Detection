from sklearn.model_selection import  cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest, RandomForestClassifier, StackingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import pandas as pd
def metrike(y_true, y_pred,model_name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred)
    confusion = confusion_matrix(y_true,y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / ( tn + fp)
    print("Accuracy: ",acc)
    print("Recall: ",recall)
    print("Precisions: ",precision)
    print("F1: ",f1)
    print("matrica: ",confusion)
    print("Specificity: ",specificity)
    auc_roc = roc_auc_score(y_true, y_pred)
    print(f'AUC-ROC: {auc_roc:.2f}')
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    metrics = {
        'Metric': ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'Confusion Matrix', 'ROC AUC','Specificity'],
        model_name: [acc, recall, precision, f1, confusion, roc_auc,specificity]
    }
    df_new = pd.DataFrame(metrics)

    filename = 'metrike.csv'

    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)

        df_final = pd.merge(df_existing, df_new, on='Metric', how='outer')
    else:

        df_final = df_new


    df_final.to_csv(filename, index=False)
    print(f"Podaci zapisani u datoteku {filename}")

def unakrsna_validacija(model,X_train,y_train):
    cv = cross_val_score(model, X_train, y_train, cv=5)
    print("Rezulati kros validacije kroz 5 iteracija:", cv)
    print("Srenja vrednost:", cv.mean())



def Stacking(X_train, X_test, y_train, y_test):
    param_grid_dt = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    param_grid_svc = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }


    grid_search_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=5)
    grid_search_dt.fit(X_train, y_train)
    best_dt = grid_search_dt.best_estimator_


    grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)
    grid_search_rf.fit(X_train, y_train)
    best_rf = grid_search_rf.best_estimator_


    grid_search_svc = GridSearchCV(SVC(probability=True), param_grid_svc, cv=5)
    grid_search_svc.fit(X_train, y_train)
    best_svc = grid_search_svc.best_estimator_


    estimators = [
        ('dt', best_dt),
        ('rf', best_rf),
        ('svc', best_svc)
    ]

    meta_model = LogisticRegression()


    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=meta_model)


    unakrsna_validacija(stacking_clf, X_train, y_train)
    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_test)
    metrike(y_test, y_pred,"Stacking")


def Bagging(X_train, X_test, y_train, y_test):

    base_model = DecisionTreeClassifier()
    unakrsna_validacija(base_model, X_train, y_train)
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(base_model, param_grid, cv=5)

    grid_search.fit(X_train, y_train)

    best_decision_tree = grid_search.best_estimator_


    bagging_clf = BaggingClassifier(estimator=best_decision_tree, n_estimators=50)

    bagging_clf.fit(X_train, y_train)
    y_pred = bagging_clf.predict(X_test)
    metrike(y_test, y_pred,"Bagging")


def Boosting(X_train, X_test, y_train, y_test):

    gradient_boosting_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    unakrsna_validacija(gradient_boosting_clf, X_train, y_train)
    gradient_boosting_clf.fit(X_train, y_train)
    y_pred = gradient_boosting_clf.predict(X_test)
    metrike(y_test, y_pred,"Boosting")



def LogistickaRegresija(X_train, X_test, y_train, y_test):
    logistic_regression = LogisticRegression()
    unakrsna_validacija(logistic_regression,X_train,y_train)
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2']  }
    grid_search = GridSearchCV(logistic_regression, param_grid, cv=5)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    metrike(y_test, y_pred,"Logisticka Regresija")


def DecisionTree(X_train, X_test, y_train, y_test):

    decision_tree = DecisionTreeClassifier()
    unakrsna_validacija(decision_tree, X_train, y_train)

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }


    grid_search = GridSearchCV(decision_tree, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    metrike(y_test,y_pred,"DecisionTree")

def odredjivanjeK(X_train, X_test, y_train, y_test):
    error_rate = []

    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 40), error_rate, color='blue', linestyle ='dashed', marker ='o',  markerfacecolor ='red', markersize = 10)
    plt.title('Greska i K vrednost ')
    plt.xlabel('K')
    plt.ylabel('Greska')


def KNN(X_train, X_test, y_train, y_test,k):
    model = KNeighborsClassifier()
    model.k = k
    unakrsna_validacija(model,X_train,y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrike(y_test,y_pred,"KNN")


