import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE

def _flatten_windows(Xw):
    return Xw.reshape(Xw.shape[0], -1)

def train_sensor_rf(Xw, yw):
    X = _flatten_windows(Xw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, yw, test_size=0.2, random_state=RANDOM_STATE, stratify=yw
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced")
    clf.fit(X_train, y_train)
    ypred = clf.predict(X_test)

    report = classification_report(y_test, ypred, digits=3)
    cm = confusion_matrix(y_test, ypred, labels=[0, 1])
    return clf, report, cm
