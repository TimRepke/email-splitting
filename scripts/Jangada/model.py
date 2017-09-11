from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scripts.utils import AnnotatedEmails, AnnotatedEmail
from scripts.utils import denotation_types
from scripts.Jangada.features import mail2features
from scripts.Jangada.cperceptron import CollinsPerceptron
import pandas as pd
import numpy as np
from pprint import pprint
from collections import Counter


def flatten(lst):
    return [l for sub in lst for l in sub]


if __name__ == '__main__':
    emails = AnnotatedEmails('/home/tim/workspace/enno/data', mail2features)
    print('loaded mails')

    X_train, X_test, X_eval = emails.features
    print('loaded features')
    y_train, y_test, y_eval = emails.five_zones_labels
    print('loaded labels')

    cp = CollinsPerceptron(5, 'model')

    #cp.fit(X_train, y_train, 40)
    print('fitted')
    y_pred = cp.predict(X_test, y_test)
    print('predicted')

    # pprint(list(zip(flatten(y_pred), flatten(y_test))))
    le = LabelEncoder()
    le.fit(AnnotatedEmail.zone_labels(5))
    a = le.transform(flatten(y_pred))
    b = le.transform(flatten(y_test))
    print(Counter(flatten(y_pred)))
    print(Counter(flatten(y_test)))
    print(le.classes_)
    print('Accuracy: ', accuracy_score(b, a))
    print(classification_report(b, a, target_names=le.classes_))
    # pprint(precision_recall_fscore_support(b, a))
    print(confusion_matrix(b, a))
