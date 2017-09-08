from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scripts.utils import AnnotatedEmails
from scripts.utils import denotation_types
from scripts.Jangada.features import mail2features
from scripts.Jangada.cperceptron import CollinsPerceptron
import pandas as pd
import numpy as np
from pprint import pprint


def flatten(lst):
    return [l for sub in lst for l in sub]


if __name__ == '__main__':
    emails = AnnotatedEmails('/home/tim/workspace/enno/data', mail2features)
    print('loaded mails')

    X_train, X_test, X_eval = emails.features
    print('loaded features')
    y_train, y_test, y_eval = emails.two_zones_labels
    print('loaded labels')

    cp = CollinsPerceptron(5, 'model')

    cp.fit(X_train, y_train, 38)
    print('fitted')
    y_pred = cp.predict(X_test, y_test)
    print('predicted')

    le = LabelEncoder()
    a = le.fit_transform(flatten(y_pred))
    b = le.fit_transform(flatten(y_test))
    print(le.classes_)
    print('Accuracy: ', accuracy_score(b, a))
    print(classification_report(b, a, target_names=le.classes_))
    # pprint(precision_recall_fscore_support(b, a))
    print(confusion_matrix(b, a))
