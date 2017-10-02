from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scripts.utils import AnnotatedEmails
from scripts.utils import denotation_types
from scripts.Zebra.features import mail2features
import pandas as pd
import numpy as np


def flatten(lst):
    return [l for sub in lst for l in sub]


def to_array(lst, cols=None):
    df = pd.DataFrame(lst, columns=cols)
    df.fillna(0, inplace=True)
    if cols is not None:
        return np.nan_to_num(df.as_matrix(cols))
    return df.columns, df.as_matrix()


if __name__ == '__main__':
    emails = AnnotatedEmails('/home/tim/workspace/enno/data', mail2features)
    print('loaded mails')

    X_train, X_test, X_eval = emails.features
    print('loaded features')
    y_train, y_test, y_eval = emails.two_zones_labels_numeric
    print('loaded labels')

    ss = StandardScaler()
    cols, X_train = to_array(flatten(X_train))
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(to_array(flatten(X_test), cols))
    X_eval = ss.transform(to_array(flatten(X_eval), cols))
    le = LabelEncoder()
    y_train = le.fit_transform(flatten(y_train))
    y_test = le.transform(flatten(y_test))
    y_eval = le.transform(flatten(y_eval))

    print('train', X_train.shape)
    print('test', X_test.shape)
    print('eval', X_eval.shape)

    xt = X_test
    yt = y_test

    svc = SVC(max_iter=200, random_state=42, verbose=1)  # , class_weight='balanced') , class_weight={0: 1.0, 1: 0.5}
    svc.fit(X_train, y_train)
    print('fitted')
    y_pred = svc.predict(xt)
    print('predicted')

    print(le.classes_)

    print('Accuracy: ', accuracy_score(yt, y_pred))
    print(classification_report(yt, y_pred))
