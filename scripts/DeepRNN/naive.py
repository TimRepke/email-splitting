from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.layers import LSTM, Activation, Dropout, Dense, Masking, GRU, Input, Embedding, Reshape
from keras import backend as K
from keras.layers.noise import GaussianNoise
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from keras.utils import np_utils, Sequence
from keras.regularizers import l2
from scripts.utils import AnnotatedEmails, AnnotatedEmail
from scripts.utils import denotation_types
from pprint import pprint
from scripts.FeatureRNN.features import mail2features
from sklearn.utils import compute_class_weight
from collections import Counter
from keras.callbacks import TensorBoard
from keras.initializers import RandomUniform
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from email import parser as ep
import tensorflow as tf
import sys
from keras.optimizers import RMSprop, SGD, Adadelta, Adagrad, Adam
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalAveragePooling1D
from keras.layers import LSTM, Lambda
from keras.layers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization

folder = "../../../../enron/data/original/"

char_index = list(' '
                  'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                  'abcdefghijklmnopqrstuvwxyz'
                  '0123456789'
                  '@€-_.:,;#\'+*~\?}=])[({/&%$§"!^°|><´`\n')
num_possible_chars = len(char_index)


def onehot(x):
    return tf.to_float(tf.one_hot(x, num_possible_chars + 1, on_value=1, off_value=0, axis=-1))


def onehot_outshape(in_shape):
    return in_shape[0], in_shape[1], num_possible_chars + 1


def char2num(c):
    return char_index.index(c) + 1 if c in char_index else 0


def mail2arr_fixed(m, one_hot=False, width=None, max_mail_len=None):
    lines = m.lines
    if width is None:
        width = max([len(l) for l in lines])
    arr = np.zeros(
        (len(lines) if max_mail_len is None else max_mail_len, width, num_possible_chars + 1)) if one_hot else \
        np.zeros((len(lines) if max_mail_len is None else max_mail_len, width))
    for i, line in enumerate(lines[:max_mail_len]):
        for j in range(len(line[:width])):
            if one_hot:
                arr[i][j][char2num(line[j])] = 1
            else:
                arr[i][j] = char2num(line[j]) + 1
    return np.nan_to_num(arr)


def mail2arr_sequenced(m, one_hot=False, max_len=None, max_mail_len=None):
    lines = m.lines
    ret = []
    for i, line in enumerate(lines[:max_mail_len]):
        ll = len(line) if max_len is None and len(line) < max_len else max_len
        if one_hot:
            tmp = np.zeros((ll, num_possible_chars + 1))
            for j in range(ll):
                tmp[j][char2num(line[j])] = 1
        else:
            tmp = np.array([char2num(c) for i, c in enumerate(line) if i < ll])
        ret.append(tmp)
    return np.nan_to_num(ret)


class MailBatches(Sequence):
    def __init__(self, mails, labels, label_encoder, batch_size, width=None, fixed_width=True, one_hot=False,
                 with_weights=True, nested=True, max_mail_len=None):
        self.batch_size = batch_size
        self.width = width
        self.fixed_width = fixed_width
        self.max_mail_len = max_mail_len
        self.mails = mails
        self.labels = labels
        self.label_encoder = label_encoder
        self.one_hot = one_hot
        self.with_weights = with_weights
        self.nested = nested
        self.lines_flat = [line for m in mails for line in m.lines]
        self.labels_flat = [label for labs in labels for label in labs]
        self.class_weights = compute_class_weight('balanced', self.label_encoder.classes_, self.labels_flat)
        print(self.class_weights)
        self.cache = {}

    def __len__(self):
        if self.nested:
            return len(self.mails) // self.batch_size
        return len(self.lines_flat) // self.batch_size

    def _get_nested(self, idx):
        mails = self.mails[idx * self.batch_size:(idx + 1) * self.batch_size]
        labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.fixed_width:
            width = self.width
            if width is None:
                width = max([len(l) for m in mails for l in m.lines])
            mails = np.array(
                [mail2arr_fixed(m, width=width, one_hot=self.one_hot, max_mail_len=self.max_mail_len) for m in mails])
        else:
            mails = np.array(
                [mail2arr_sequenced(m, one_hot=self.one_hot, max_len=self.width, max_mail_len=self.max_mail_len) for m
                 in mails])

        # print(mails.shape)
        # print([mm.shape for mm in mails])
        # print(mails[0][0].shape)
        # print(labels[0][0])
        weights = [np.array([self.class_weights[self.label_encoder.transform([l])[0]]
                             for l in labs[:self.max_mail_len]])
                   for labs in labels]
        weights = pad_sequences(weights)
        mails = pad_sequences(mails)
        labels = [self._encode_labels(labs[:self.max_mail_len]) for labs in labels]
        labels = pad_sequences(labels)

        return mails, labels, weights

    def _get_flat(self, idx):
        lines_in = self.lines_flat[idx * self.batch_size:(idx + 1) * self.batch_size]
        labels = self.labels_flat[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.fixed_width:
            width = self.width
            if width is None:
                width = max([len(l) for l in lines_in])
            if self.one_hot:
                lines = np.zeros((len(lines_in), width, num_possible_chars + 1))
                for i, line in enumerate(lines_in):
                    for j, c in enumerate(line[:width]):
                        lines[i][j][char2num(c)] = 1
            else:
                lines = np.zeros((len(lines_in), width))
                for i, line in enumerate(lines_in):
                    for j, c in enumerate(line[:width]):
                        lines[i][j] = char2num(c)
        else:
            if self.one_hot:
                lines = []
                for i, line in enumerate(lines_in):
                    tmp = np.zeros((len(line), num_possible_chars + 1))
                    for j, c in enumerate(line if self.width is None else line[:self.width]):
                        tmp[j][char2num(c)] = 1
                    lines.append(tmp)
            else:
                lines = []
                for i, line in enumerate(lines_in):
                    lines.append(np.array([char2num(c) for c in (line if self.width is None else line[:self.width])]))
            lines = np.array(lines)

        # print(labels)
        labs = self.label_encoder.transform(labels)
        weights = np.array([self.class_weights[l] for l in labs])
        labels = np_utils.to_categorical(labs, len(self.label_encoder.classes_))

        return lines, labels, weights

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise StopIteration()
        if idx in self.cache:
            return self.cache[idx]

        if self.nested:
            mails, labels, weights = self._get_nested(idx)
        else:
            mails, labels, weights = self._get_flat(idx)

        if self.with_weights:
            self.cache[idx] = (mails, labels, weights)
            return mails, labels, weights
        else:
            self.cache[idx] = (mails, labels)
            return mails, labels

    def _encode_labels(self, lst):
        return np_utils.to_categorical(self.label_encoder.transform(lst),
                                       len(self.label_encoder.classes_))

    def materialise(self):
        x = []
        y = []
        w = []
        for i in range(self.__len__()):
            xi, yi, wi = self.__getitem__(i)
            x.append(xi)
            y.append(yi)
            w.append(wi)

        if not self.nested:
            x = np.concatenate(x, axis=0)
            y = np.concatenate(y, axis=0)
            w = np.concatenate(w, axis=0)

        return (x, y, w) if self.with_weights else (x, y)

    def on_epoch_end(self):
        pass


def flatten(lst):
    return [l for sub in lst for l in sub]


if __name__ == "__main__":
    zones = 5
    batch_size = 5
    epochs = 2
    max_line_len = 80
    max_mail_len = None
    fixed_width = True
    as_onehot = True
    nested = True
    predict_on = None  # 30

    labels = AnnotatedEmail.zone_labels(zones)
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    emails = AnnotatedEmails('/home/tim/workspace/enno/data', lambda m: m)
    print('loaded mails')

    train_mails, test_mails, eval_mails = emails.features
    print('loaded texts')

    if zones == 5:
        y_train, y_test, y_eval = emails.five_zones_labels
    elif zones == 3:
        y_train, y_test, y_eval = emails.three_zones_labels
    else:
        y_train, y_test, y_eval = emails.two_zones_labels
    print('loaded labels')

    in_size = num_possible_chars + 1
    out_size = len(label_encoder.classes_)
    print(out_size, label_encoder.classes_)

    train = MailBatches(train_mails, y_train, label_encoder, batch_size, max_mail_len=max_mail_len,
                        width=max_line_len, fixed_width=fixed_width, one_hot=as_onehot, nested=nested)

    test = MailBatches(test_mails, y_test, label_encoder, batch_size,
                       width=max_line_len, fixed_width=fixed_width, one_hot=as_onehot, nested=nested)

    # callbacks = [TensorBoard(log_dir='./logs/' + time.strftime('%Y-%m-%d_%H-%M'), write_images=True, histogram_freq=1)]
    # callbacks = None
    val = test.materialise()
    if nested:
        val = (val[0][0], val[1][0], val[2][0])

    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(max_line_len, num_possible_chars + 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    # model.add(Dropout(0.4))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(out_size, activation='softmax'))

    # model.add(Bidirectional(GRU(32, return_sequences=True)))
    mail_model = Sequential()
    mail_model.add(TimeDistributed(model, input_shape=(None, max_line_len, num_possible_chars + 1)))
    mail_model.add(GRU(out_size,
                       activation='softmax', return_sequences=True))
    mail_model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    mail_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop',
                       sample_weight_mode='temporal',
                       metrics=['accuracy'])

    num_batches = len(train)
    inner_batch_size = 30
    K.set_learning_phase(1)
    for epoch in range(epochs):
        for batch, (batch_x, batch_y, batch_w) in enumerate(train):
            print(epoch+1, '-', str(batch+1) + '/' + str(num_batches+1), ')')
            # print(batch_x)
            xf = flatten(batch_x)
            yf = flatten(batch_y)
            wf = flatten(batch_w)
            for inner_batch in range(len(xf) // inner_batch_size):
                model.train_on_batch(np.array(xf[inner_batch * inner_batch_size:(inner_batch + 1) * inner_batch_size]),
                                     np.array(yf[inner_batch * inner_batch_size:(inner_batch + 1) * inner_batch_size]),
                                     sample_weight=np.array(
                                         wf[inner_batch * inner_batch_size:(inner_batch + 1) * inner_batch_size]))
#           loss_a = model.test_on_batch(val[0], val[1], val[2])
            if epoch > 0:
                mail_model.train_on_batch(batch_x, batch_y, sample_weight=batch_w)
    K.set_learning_phase(0)

    test = MailBatches(test_mails, y_test, label_encoder, len(test_mails),
                       width=max_line_len, fixed_width=fixed_width, one_hot=as_onehot,
                       with_weights=False, nested=nested)
    Y_pred = mail_model.predict_generator(test, steps=len(test))
    print(Y_pred.shape)

    y_pred = []
    y_pred_p = []

    if nested:
        a = not True
        for m, mm in zip(Y_pred, test_mails):
            if a:
                a = False
                print(m.shape, len(mm))
                for w, l in zip(m[0:len(mm)], emails.test_set[0].lines):
                    print(list(w), label_encoder.inverse_transform([w.argmax()])[0], ')', l[:100])
                    # print(list(m)[0:len(mm)])
            # y_pred += list(m.argmax(axis=1))[len(m)-len(mm):]  # [0:len(mm)]
            # y_pred_p += list(m)[len(m)-len(mm):]  # [0:len(mm)]
            y_pred += list(m.argmax(axis=1))[:len(mm)]
            y_pred_p += list(m)[:len(mm)]
    else:
        y_pred = Y_pred.argmax(axis=1)
        y_pred_p = Y_pred

    # a = le.transform(flatten(y_test))
    a = flatten(y_test)[:len(y_pred)]
    b = label_encoder.inverse_transform(y_pred)

    print(len(a))
    print(len(b))

    # pprint(list(zip(a, b)))
    # pprint(list(zip(a, y_pred_p)))

    print('Accuracy: ', accuracy_score(a, b))
    print(classification_report(a, b, target_names=label_encoder.classes_))
    # pprint(precision_recall_fscore_support(b, a))
    print(label_encoder.classes_)
    print(confusion_matrix(a, b, labels=label_encoder.classes_))
