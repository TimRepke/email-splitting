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
from email import parser as ep
import tensorflow as tf
import sys
from keras.optimizers import RMSprop, SGD, Adadelta, Adagrad, Adam
from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalAveragePooling1D, Highway
from keras.layers import LSTM, Lambda
from keras.layers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras_contrib.layers import CRF

folder = "../../../../enron/data/original/"
# folder = "../../data/asf/annotated/"

char_index = list(' '
                  'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                  'abcdefghijklmnopqrstuvwxyz'
                  '0123456789'
                  '@€-_.:,;#\'+*~\?}=])[({/&%$§"!^°|><´`\n')
num_possible_chars = len(char_index)


def char2num(c):
    return char_index.index(c) + 1 if c in char_index else 0


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


def flatten(lst):
    return [l for sub in lst for l in sub]


def get_conv_model(emb_size, le):
    n_labels = len(le.classes_)
    in_line = Input(shape=(None, num_possible_chars + 1), dtype='float32')
    # hw = TimeDistributed(Highway())(in_line)
    # hw = TimeDistributed(Highway())(hw)
    # hw = TimeDistributed(Highway())(hw)
    # embedding_conv = Conv1D(64, 3, activation='relu')(hw)
    embedding_conv = Conv1D(64, 3, activation='relu')(in_line)
    # embedding_conv = Conv1D(64, 3, activation='relu')(embedding_conv)
    embedding_conv = MaxPooling1D(3)(embedding_conv)
    embedding_conv = Conv1D(128, 3, activation='relu')(embedding_conv)
    embedding_conv = GlobalAveragePooling1D()(embedding_conv)

    embedding = Dropout(0.4)(embedding_conv)
    embedding = Dense(emb_size)(embedding)

    output = Dense(n_labels, activation='softmax')(embedding)

    model = Model(inputs=in_line, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # model.summary()

    return model


def get_rnn_model(emb_size, le):
    n_labels = len(le.classes_)
    in_line = Input(shape=(None, num_possible_chars + 1), dtype='float32')
    in_line = Masking()(in_line)

    embedding = GRU(emb_size, return_sequences=False)(in_line)
    output = Dense(n_labels, activation='softmax')(embedding)
    model = Model(inputs=in_line, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # model.summary()

    return model


def get_line_training_sets(validation_size, label_encoder, line_len):
    # label_encoder = line_label_encoder
    y_train, y_test, y_eval = get_labels(len(label_encoder.classes_))
    train = LineBatches(train_mails, y_train, label_encoder, fix_to_max=True,
                        batch_size=line_training_batch_size, max_len=line_len, one_hot=True, with_weights=True)

    test = LineBatches(test_mails, y_test, label_encoder, fix_to_max=True,
                       batch_size=validation_size, max_len=line_len, one_hot=True, with_weights=True)
    val = test.__getitem__(0)

    test = LineBatches(test_mails, y_test, label_encoder, fix_to_max=True,
                       batch_size=line_training_batch_size, max_len=line_len, one_hot=True, with_weights=False)

    return train, val, test, y_test


def get_labels(zones):
    if zones == 2:
        return emails.two_zones_labels
    if zones == 3:
        return emails.three_zones_labels
    return emails.five_zones_labels


def evaluate(yt, yp, labels):
    print(len(yt))
    print(len(yp))

    print('Accuracy: ', accuracy_score(yt, yp))
    print(classification_report(yt, yp, target_names=labels))
    print(labels)
    print(confusion_matrix(yt, yp, labels=labels))


def eval_line_training(Y_pred, y_test, le):
    print(Y_pred.shape)

    y_pred = Y_pred.argmax(axis=1)
    print(y_pred)

    evaluate(flatten(y_test)[:len(y_pred)],
             le.inverse_transform(y_pred),
             le.classes_)


def get_label_encoder(num_zones):
    label_encoder = LabelEncoder()
    label_encoder.fit(AnnotatedEmail.zone_labels(num_zones))
    return label_encoder


if __name__ == "__main__":
    line_training_batch_size = 20
    emails = AnnotatedEmails('/home/tim/workspace/enno/data', lambda m: m)
    print('loaded mails')

    train_mails, test_mails, eval_mails = emails.features
    print('loaded texts')

    for chars_per_line in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]:
        for num_labels in [2, 5]:
            for embedding_size in [16, 32, 64]:
                print("===================================================")
                print("line len: %d, num labels: %d, embedding size: %d" % (chars_per_line, num_labels, embedding_size))
                lab_encoder = get_label_encoder(num_labels)
                train, val, test, y_test = get_line_training_sets(100, lab_encoder, chars_per_line)

                print("Training conv model")
                line_model = get_conv_model(embedding_size, lab_encoder)
                history = line_model.fit_generator(train,
                                                   steps_per_epoch=len(train),
                                                   epochs=5,
                                                   verbose=0,
                                                   validation_data=val,
                                                   validation_steps=None).history
                Y_pred = line_model.predict_generator(test, steps=len(test))
                print(history)
                eval_line_training(Y_pred, y_test, lab_encoder)

                print("Training rnn model")
                line_model = get_conv_model(embedding_size, lab_encoder)
                history = line_model.fit_generator(train,
                                                   steps_per_epoch=len(train),
                                                   epochs=5,
                                                   verbose=0,
                                                   validation_data=val,
                                                   validation_steps=None).history
                Y_pred = line_model.predict_generator(test, steps=len(test))
                print(history)
                eval_line_training(Y_pred, y_test, lab_encoder)

