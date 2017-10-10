from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.layers import LSTM, Activation, Dropout, Dense, Masking, GRU, Input, Embedding, Reshape
import keras.backend as K
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


class LineBatches(Sequence):
    def on_epoch_end(self):
        pass

    def __init__(self, mails, labels, label_encoder, batch_size,
                 max_len=None, one_hot=True, with_weights=True, fix_to_max=False):
        self.lines = [line for m in mails for line in m.lines]
        self.labels = [label for labs in labels for label in labs]
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.max_len = max_len
        self.fix_to_max = fix_to_max
        self.one_hot = one_hot
        self.with_weights = with_weights
        self.class_weights = compute_class_weight('balanced', self.label_encoder.classes_, self.labels)
        print(self.class_weights)
        self.cache = {}

    def __len__(self):
        return len(self.lines) // self.batch_size

    def __getitem__(self, idx):
        lines = self.lines[idx * self.batch_size:(idx + 1) * self.batch_size]
        labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        longest_line = max([len(line) for line in lines])
        if self.max_len is not None and (longest_line > self.max_len or self.fix_to_max):
            longest_line = self.max_len

        x = np.zeros((self.batch_size, longest_line, num_possible_chars + 1)) if self.one_hot else \
            np.zeros((self.batch_size, longest_line))
        for i, line in enumerate(lines):
            for j, c in enumerate(line):
                if j >= longest_line:
                    break
                if self.one_hot:
                    x[i][j][char2num(c)] = 1
                else:
                    x[i][j] = char2num(c)

        y = self.label_encoder.transform(labels)
        w = np.array([self.class_weights[l] for l in y])
        y = np_utils.to_categorical(y, len(self.label_encoder.classes_))

        return (x, y, w) if self.with_weights else (x, y)

    def materialise(self):
        x = []
        y = []
        w = []
        for i in range(self.__len__()):
            xi, yi, wi = self.__getitem__(i)
            x.append(xi)
            y.append(yi)
            w.append(wi)

        return (x, y, w) if self.with_weights else (x, y)


class MailBatches(Sequence):
    def on_epoch_end(self):
        pass

    def __init__(self, mails, labels, label_encoder, batch_size, embedding_function=None,
                 max_len=None, one_hot=True, with_weights=True, fix_to_max=False):
        self.embedding_function = embedding_function
        self.mails = mails
        self.labels = labels
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.max_len = max_len
        self.fix_to_max = fix_to_max
        self.one_hot = one_hot
        self.with_weights = with_weights
        self.class_weights = compute_class_weight('balanced', self.label_encoder.classes_, flatten(labels))
        print(self.class_weights)
        self.cache = {}

    def __len__(self):
        return len(self.mails) // self.batch_size

    def __getitem__(self, idx):
        mails = self.mails[idx * self.batch_size:(idx + 1) * self.batch_size]
        labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        longest_line = max([len(l) for m in mails for l in m.lines])
        if self.max_len is not None and (longest_line > self.max_len or self.fix_to_max):
            longest_line = self.max_len
        longest_mail = max([len(m.lines) for m in mails])

        if self.embedding_function is not None:
            x = np.zeros((self.batch_size, longest_mail, line_embedding_size))
        elif self.one_hot:
            x = np.zeros((self.batch_size, longest_mail, longest_line, num_possible_chars + 1))
        else:
            x = np.zeros((self.batch_size, longest_mail, longest_line))

        for i, mail in enumerate(mails):
            xi = np.zeros((longest_mail, longest_line, num_possible_chars+1)) if self.one_hot else \
                np.zeros((longest_mail, longest_line))
            for j, line in enumerate(mail.lines):
                for k, c in enumerate(line):
                    if k >= longest_line:
                        break
                    if self.one_hot:
                        xi[j][k][char2num(c)] = 1
                    else:
                        xi[j][k] = char2num(c)
            if self.embedding_function is not None:
                xi = self.embedding_function(xi)
            x[i] = xi

        w = np.zeros((self.batch_size, longest_mail))
        y = np.zeros((self.batch_size, longest_mail, len(self.label_encoder.classes_)))
        for i, m_labels in enumerate(labels):
            labels_encoded = self.label_encoder.transform(m_labels)
            for j, label in enumerate(labels_encoded):
                w[i][j] = self.class_weights[label]
                y[i][j][label] = 1

        return (x, y, w) if self.with_weights else (x, y)

    def materialise(self):
        x = []
        y = []
        w = []
        for i in range(self.__len__()):
            xi, yi, wi = self.__getitem__(i)
            x.append(xi)
            y.append(yi)
            w.append(wi)

        return (x, y, w) if self.with_weights else (x, y)


def flatten(lst):
    return [l for sub in lst for l in sub]


def get_line_model(embedding_size):
    in_line = Input(shape=(None, num_possible_chars + 1), dtype='float32')

    embedding_conv = Conv1D(64, 3, activation='relu')(in_line)
    # embedding_conv = Conv1D(64, 3, activation='relu')(embedding_conv)
    embedding_conv = MaxPooling1D(3)(embedding_conv)
    embedding_conv = Conv1D(128, 3, activation='relu')(embedding_conv)
    embedding_conv = GlobalAveragePooling1D()(embedding_conv)

    embedding = Dropout(0.4)(embedding_conv)
    embedding = Dense(embedding_size)(embedding)

    output = Dense(2, activation='softmax')(embedding)

    model = Model(inputs=in_line, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    embedding_lamda = lambda_embedding(model, embedding)

    return model, embedding_lamda


def get_line_training_sets(validation_size):
    y_train, y_test, y_eval = emails.two_zones_labels
    train = LineBatches(train_mails, y_train, label_encoder_two, fix_to_max=True,
                        batch_size=line_training_batch_size, max_len=max_line_len, one_hot=True, with_weights=True)

    test = LineBatches(test_mails, y_test, label_encoder_two, fix_to_max=True,
                       batch_size=validation_size, max_len=max_line_len, one_hot=True, with_weights=True)
    val = test.__getitem__(0)

    test = LineBatches(test_mails, y_test, label_encoder_two, fix_to_max=True,
                       batch_size=line_training_batch_size, max_len=max_line_len, one_hot=True, with_weights=False)

    return train, val, test, y_test


def get_mail_training_sets(validation_size, test_size, label_encoder, embedding_function):
    if len(label_encoder.classes_) == 2:
        y_train, y_test, y_eval = emails.two_zones_labels
    else:
        y_train, y_test, y_eval = emails.five_zones_labels

    train = MailBatches(train_mails, y_train, label_encoder, batch_size=mail_two_zone_batch_size,
                        embedding_function=embedding_function, fix_to_max=True,
                        max_len=max_line_len, one_hot=True, with_weights=True)

    test = MailBatches(test_mails, y_test, label_encoder, batch_size=validation_size,
                       embedding_function=embedding_function, fix_to_max=True,
                       max_len=max_line_len, one_hot=True, with_weights=True)
    val = test.__getitem__(0)

    test = MailBatches(test_mails, y_test, label_encoder, batch_size=test_size,
                       embedding_function=embedding_function, fix_to_max=True,
                       max_len=max_line_len, one_hot=True, with_weights=False)

    return train, val, test, y_test


def eval(yp, yt, labels):
    print(len(yt))
    print(len(yp))

    print('Accuracy: ', accuracy_score(yt, yp))
    print(classification_report(yt, yp, target_names=labels))
    print(labels)
    print(confusion_matrix(yt, yp, labels=labels))


def eval_line_training(Y_pred, y_test):
    print(Y_pred.shape)

    y_pred = Y_pred.argmax(axis=1)

    eval(flatten(y_test)[:len(y_pred)],
         label_encoder_two.inverse_transform(y_pred),
         label_encoder_two.classes_)


def eval_mail_training(Y_pred, y_test, le):
    print(Y_pred.shape)
    y_pred = []
    y_pred_p = []
    for yp, yt in zip(Y_pred, y_test):
        y_pred += list(yp.argmax(axis=1))[:len(yt)]
        y_pred_p += list(yp)[:len(yt)]

    eval(flatten(y_test)[:len(y_pred)],
         le.inverse_transform(y_pred),
         le.classes_)


def get_label_encoder(num_zones):
    label_encoder = LabelEncoder()
    label_encoder.fit(AnnotatedEmail.zone_labels(num_zones))
    return label_encoder


def lambda_embedding(embedding_model, embedding_layer):
    model_in = [embedding_model.input]
    embedding_func = K.function(model_in + [K.learning_phase()], [embedding_layer])

    def lambdo(x):
        return embedding_func([x, 0.])[0]

    return lambdo


def get_mail_model():
    in_mail = Input(shape=(None, line_embedding_size), dtype='float32')

    hidden = LSTM(32,
                  return_sequences=True,
                  implementation=0)(in_mail)
    output = LSTM(2,
                  return_sequences=True,
                  activation='softmax',
                  implementation=0)(hidden)

    model = Model(inputs=in_mail, outputs=output)

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.01),
                  sample_weight_mode='temporal',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    line_training_batch_size = 20
    line_training_epochs = 2
    line_embedding_size = 32

    mail_two_zone_batch_size = 2
    mail_two_zone_epochs = 2

    zones = 5
    max_line_len = 100

    label_encoder_two = get_label_encoder(2)
    label_encoder_five = get_label_encoder(5)

    emails = AnnotatedEmails('/home/tim/workspace/enno/data', lambda m: m)
    print('loaded mails')

    train_mails, test_mails, eval_mails = emails.features
    print('loaded texts')

    # Training line embeddings
    line_model, line_embedding = get_line_model(line_embedding_size)
    train, val, test, y_test = get_line_training_sets(100)
    history = line_model.fit_generator(train,
                                       steps_per_epoch=len(train),
                                       epochs=line_training_epochs,
                                       verbose=1,
                                       validation_data=val,
                                       validation_steps=None).history
    Y_pred = line_model.predict_generator(test, steps=len(test))
    eval_line_training(Y_pred, y_test)

    # Training two zone model
    two_zone_mail_model = get_mail_model()
    train, val, test, y_test = get_mail_training_sets(100, 50, label_encoder_two, line_embedding)
    history = two_zone_mail_model.fit_generator(train,
                                                steps_per_epoch=len(train),
                                                epochs=mail_two_zone_epochs,
                                                verbose=1,
                                                validation_data=val,
                                                validation_steps=None).history
    Y_pred = two_zone_mail_model.predict_generator(test, steps=1, verbose=1)
    eval_mail_training(Y_pred, y_test, label_encoder_two)
