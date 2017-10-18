from sklearn.svm import LinearSVC, SVC
import sys

sys.path.insert(0, '/home/tim/Uni/HPI/workspace/email-splitting')
sys.path.insert(0, '/home/tim/Uni/HPI/workspace/email-splitting/scripts')
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
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalAveragePooling1D, Highway
from keras.layers import LSTM, Lambda
from keras.layers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras_contrib.layers import CRF

# folder = "../../../../enron/data/original/"
# folder = "../../data/asf/annotated/"

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

    def __init__(self, mails, labels_, label_encoder, batch_size,
                 max_len=None, one_hot=True, with_weights=True, fix_to_max=False):
        self.lines = [line for m in mails for line in m.lines]
        self.labels = [label for labs in labels_ for label in labs]
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.max_len = max_len
        self.fix_to_max = fix_to_max
        self.one_hot = one_hot
        self.with_weights = with_weights
        self.class_weights = compute_class_weight('balanced', self.label_encoder.classes_, self.labels)

        tmp = mails[0]
        tmp.perturbation = 0.15
        print('boosting lines by', len(self.lines) // 3)
        for li, la in zip(self.lines[:len(self.lines) // 3], self.labels[:len(self.labels) // 3]):
            self.lines = [tmp._perturbate_line(li)] + self.lines
            self.labels = [la] + self.labels
        tmp.perturbation = 0.0

        # print(self.class_weights)
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

    def __init__(self, mails, labels, label_encoder, batch_size, embedding_functions=None,
                 max_len=None, one_hot=True, with_weights=True, fix_to_max=False):
        self.embedding_functions = embedding_functions
        self.mails = mails
        self.labels = labels
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.max_len = max_len
        self.fix_to_max = fix_to_max
        self.one_hot = one_hot
        self.with_weights = with_weights
        self.class_weights = compute_class_weight('balanced', self.label_encoder.classes_, flatten(labels))
        # print(self.class_weights)
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

        if self.embedding_functions is not None:
            x = np.zeros((self.batch_size, longest_mail, line_embedding_size * len(self.embedding_functions)))
        elif self.one_hot:
            x = np.zeros((self.batch_size, longest_mail, longest_line, num_possible_chars + 1))
        else:
            x = np.zeros((self.batch_size, longest_mail, longest_line))

        for i, mail in enumerate(mails):
            xi = np.zeros((longest_mail, longest_line, num_possible_chars + 1)) if self.one_hot else \
                np.zeros((longest_mail, longest_line))
            for j, line in enumerate(mail.lines):
                for k, c in enumerate(line):
                    if k >= longest_line:
                        break
                    if self.one_hot:
                        xi[j][k][char2num(c)] = 1
                    else:
                        xi[j][k] = char2num(c)
            if self.embedding_functions is not None:
                xi = np.concatenate([ef(xi) for ef in self.embedding_functions], axis=1)
                # set padded lines to zero, embedding will not do that!
                for li in range(len(mail.lines), longest_mail):
                    xi[li] = np.zeros((line_embedding_size * len(self.embedding_functions),))

            x[i] = xi

        y = np.zeros((self.batch_size, longest_mail, len(self.label_encoder.classes_)))
        w = np.zeros((self.batch_size, longest_mail))
        for i, m_labels in enumerate(labels):
            labels_encoded = self.label_encoder.transform(m_labels)
            for j, label in enumerate(labels_encoded):
                y[i][j][label] = 1
                w[i][j] = self.class_weights[label]

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


def get_line_model(embedding_size, le):
    num_labels = len(le.classes_)
    in_line = Input(shape=(None, num_possible_chars + 1), dtype='float32')
    embedding_conv = Conv1D(64, 3, activation='relu')(in_line)
    embedding_conv = MaxPooling1D(3)(embedding_conv)
    embedding_conv = Conv1D(128, 3, activation='relu')(embedding_conv)
    embedding_conv = GlobalAveragePooling1D()(embedding_conv)

    embedding = Dropout(0.4)(embedding_conv)
    embedding = Dense(embedding_size)(embedding)

    output = Dense(num_labels, activation='softmax')(embedding)

    model = Model(inputs=in_line, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    embedding_lamda = lambda_embedding(model, embedding)

    return model, embedding_lamda


def get_line_training_sets(validation_size, label_encoder):
    # label_encoder = line_label_encoder
    y_train, y_test, y_eval = get_labels(len(label_encoder.classes_))
    train = LineBatches(train_mails, y_train, label_encoder, fix_to_max=True,
                        batch_size=line_training_batch_size, max_len=max_line_len, one_hot=True, with_weights=True)

    test = LineBatches(test_mails, y_test, label_encoder, fix_to_max=True,
                       batch_size=validation_size, max_len=max_line_len, one_hot=True, with_weights=True)
    val = test.__getitem__(0)

    test = LineBatches(test_mails, y_test, label_encoder, fix_to_max=True,
                       batch_size=line_training_batch_size, max_len=max_line_len, one_hot=True, with_weights=False)

    return train, val, test, y_test


def get_labels(zones):
    if zones == 2:
        return emails.two_zones_labels
    if zones == 3:
        return emails.three_zones_labels
    return emails.five_zones_labels


def get_mail_training_sets(validation_size, test_size, efuncs):
    label_encoder = mail_label_encoder
    y_train_, y_test_, y_eval_ = get_labels(len(label_encoder.classes_))
    use_weights = False
    train_ = MailBatches(train_mails, y_train_, label_encoder, batch_size=mail_two_zone_batch_size,
                         embedding_functions=efuncs, fix_to_max=True,
                         max_len=max_line_len, one_hot=True, with_weights=use_weights)

    test_ = MailBatches(test_mails, y_test_, label_encoder, batch_size=validation_size,
                        embedding_functions=efuncs, fix_to_max=True,
                        max_len=max_line_len, one_hot=True, with_weights=use_weights)
    val_ = test_.__getitem__(0)

    test_ = MailBatches(test_mails, y_test_, label_encoder, batch_size=test_size,
                        embedding_functions=efuncs, fix_to_max=True,
                        max_len=max_line_len, one_hot=True, with_weights=False)

    return train_, val_, test_, y_test_


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


def eval_mail_training(Y_pred, y_test):
    le = mail_label_encoder
    print(Y_pred.shape)
    y_pred = []
    y_pred_p = []
    for yp, yt in zip(Y_pred, y_test):
        y_pred += list(yp.argmax(axis=1))[:len(yt)]
        y_pred_p += list(yp)[:len(yt)]

    evaluate(flatten(y_test)[:len(y_pred)],
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


def get_mail_model(bidir=True):
    output_size = len(mail_label_encoder.classes_)
    in_mail = Input(shape=(None, line_embedding_size * (1 if n_zones == 2 else 2)), dtype='float32')

    mask = Masking()(in_mail)
    if bidir:
        hidden = Bidirectional(GRU(32 // 2,
                                   return_sequences=True,
                                   implementation=0))(mask)
    else:
        hidden = GRU(32,
                     return_sequences=True,
                     implementation=0)(mask)
    crf = CRF(output_size, sparse_target=False)
    output = crf(hidden)

    model = Model(inputs=in_mail, outputs=output)

    # model.summary()

    model.compile(loss=crf.loss_function,  # 'categorical_crossentropy',
                  optimizer=Adam(lr=0.01),
                  # sample_weight_mode='temporal',
                  metrics=[crf.accuracy])  # metrics=['accuracy'])
    return model


def get_5zone_model(n_trainsamples):
    # Training line embeddings
    line_model_a, line_embedding_a = get_line_model(line_embedding_size, label_encoder_five)
    train, val, test, y_test = get_line_training_sets(50, label_encoder_five)
    history = line_model_a.fit_generator(train,
                                         steps_per_epoch=len(train),
                                         epochs=5,
                                         verbose=0,
                                         validation_data=val,
                                         validation_steps=None).history
    # Y_pred = line_model_a.predict_generator(test, steps=len(test))
    print('fitted embedding a (five zones)')
    print(history)
    # eval_line_training(Y_pred, y_test, label_encoder_five)

    line_model_b, line_embedding_b = get_line_model(line_embedding_size, label_encoder_two)
    train, val, test, y_test = get_line_training_sets(50, label_encoder_two)
    history = line_model_b.fit_generator(train,
                                         steps_per_epoch=len(train),
                                         epochs=2,
                                         verbose=0,
                                         validation_data=val,
                                         validation_steps=None).history
    # Y_pred = line_model_b.predict_generator(test, steps=len(test))
    print('fitted embedding b (two zones)')
    print(history)
    # eval_line_training(Y_pred, y_test, label_encoder_two)

    # Training two zone model
    two_zone_mail_model = get_mail_model(bidir=False)
    train, val, test, y_test = get_mail_training_sets(50, 50, [line_embedding_a, line_embedding_b])

    n = int(len(train) * n_trainsamples)
    print('>>>>> N train epochs', n, '(', n * mail_two_zone_batch_size, ')')
    history = two_zone_mail_model.fit_generator(train,
                                                steps_per_epoch=n,
                                                epochs=mail_two_zone_epochs,
                                                verbose=0,
                                                validation_data=val,
                                                validation_steps=None).history
    print('fitted model')
    print(history)
    return two_zone_mail_model, [line_embedding_a, line_embedding_b]


def get_2zone_model(n_trainsamples):
    # Training line embeddings
    line_model_b, line_embedding_b = get_line_model(line_embedding_size, label_encoder_two)
    train, val, test, y_test = get_line_training_sets(50, label_encoder_two)
    history = line_model_b.fit_generator(train,
                                         steps_per_epoch=len(train),
                                         epochs=2,
                                         verbose=0,
                                         validation_data=val,
                                         validation_steps=None).history
    # Y_pred = line_model_b.predict_generator(test, steps=len(test))
    print('fitted embedding b (two zones)')
    print(history)
    # eval_line_training(Y_pred, y_test, label_encoder_two)

    # Training two zone model
    two_zone_mail_model = get_mail_model()
    train, val, test, y_test = get_mail_training_sets(50, 50, [line_embedding_b])

    n = int(len(train) * n_trainsamples)
    print('>>>>> N train epochs', n, '/', len(train), '(', n * mail_two_zone_batch_size, ')')
    history = two_zone_mail_model.fit_generator(train,
                                                steps_per_epoch=n,
                                                epochs=mail_two_zone_epochs,
                                                verbose=0,
                                                validation_data=val,
                                                validation_steps=None).history
    print('fitted model')
    print(history)
    return two_zone_mail_model, [line_embedding_b]


if __name__ == "__main__":
    line_training_batch_size = 20
    line_training_epochs = 2
    line_embedding_size = 32

    mail_two_zone_batch_size = 2
    mail_two_zone_epochs = 4

    max_line_len = 80

    label_encoder_two = get_label_encoder(2)
    label_encoder_five = get_label_encoder(5)

    for to in [0, 1]:
        for n_zones in [2, 5]:
            train_on = ["../../data/enron/annotated/", "../../data/asf/annotated/"][to]
            test_on = ["../../data/enron/annotated/", "../../data/asf/annotated/"][(to + 1) % 2]
            print('=============================================================')
            print('ZONES:', n_zones)
            emails = AnnotatedEmails(train_on, lambda m: m)
            print('loaded mails')

            train_mails, test_mails, eval_mails = emails.features
            print('loaded texts')

            line_label_encoder = label_encoder_two
            print('train on:', train_on)
            print('test on:', test_on)
            label_encoder_mails = get_label_encoder(n_zones)
            mail_label_encoder = label_encoder_mails
            setsize = 1.0
            # for setsize in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            print('-------------------------------')
            print('SET SIZE (TRAIN):', setsize)

            if n_zones == 2:
                mail_model, embed_funcs = get_2zone_model(setsize)
            else:
                mail_model, embed_funcs = get_5zone_model(setsize)

            emails_ = AnnotatedEmails(test_on, lambda m: m)
            print('loaded mails')

            _, test_mails, eval_mails = emails_.features
            print('loaded features')

            if len(label_encoder_mails.classes_) == 5:
                _, y_test, y_eval = emails_.five_zones_labels
            else:
                _, y_test, y_eval = emails_.two_zones_labels

            le = label_encoder_mails
            labels = le.classes_
            class_weights = compute_class_weight('balanced', le.classes_, flatten(y_test))
            print('loaded labels')

            test = MailBatches(test_mails, y_test, le, batch_size=len(y_test),
                               embedding_functions=embed_funcs, fix_to_max=True,
                               max_len=max_line_len, one_hot=True, with_weights=False)
            eval = MailBatches(eval_mails, y_eval, le, batch_size=len(y_eval),
                               embedding_functions=embed_funcs, fix_to_max=True,
                               max_len=max_line_len, one_hot=True, with_weights=False)

            print('----------- TEST ---------------')
            # xt, _ = test.__getitem__(0)
            print('..')
            Y_pred = mail_model.predict_generator(test, steps=1, verbose=0)
            # Y_pred = two_zone_mail_model.predict(xt, verbose=0)
            y_pred = []
            y_pred_p = []
            for yp, yt in zip(Y_pred, y_test):
                y_pred += list(yp.argmax(axis=1))[:len(yt)]
                y_pred_p += list(yp)[:len(yt)]

            yt = flatten(y_test)[:len(y_pred)]
            yp = le.inverse_transform(y_pred)

            print('Accuracy: ', accuracy_score(yt, yp))
            print(classification_report(yt, yp, target_names=labels))
            print('Accuracy (weighted): ', accuracy_score(yt, yp,
                                                          sample_weight=[class_weights[s] for s in
                                                                         le.transform(yt)]))
            print(classification_report(yt, yp, target_names=le.classes_,
                                        sample_weight=[class_weights[s] for s in le.transform(yt)]))
            print(labels)
            print(confusion_matrix(yt, yp, labels=labels))

            print('----------- EVAL ---------------')
            # xt, _ = eval.__getitem__(0)
            print('..')
            Y_pred = mail_model.predict_generator(eval, steps=1, verbose=0)
            # Y_pred = two_zone_mail_model.predict(xt, verbose=0)
            y_pred = []
            y_pred_p = []
            for yp, yt in zip(Y_pred, y_eval):
                y_pred += list(yp.argmax(axis=1))[:len(yt)]
                y_pred_p += list(yp)[:len(yt)]

            yt = flatten(y_eval)[:len(y_pred)]
            yp = le.inverse_transform(y_pred)

            print('Accuracy: ', accuracy_score(yt, yp))
            print(classification_report(yt, yp, target_names=labels))
            print('Accuracy (weighted): ', accuracy_score(yt, yp,
                                                          sample_weight=[class_weights[s] for s in
                                                                         le.transform(yt)]))
            print(classification_report(yt, yp, target_names=le.classes_,
                                        sample_weight=[class_weights[s] for s in le.transform(yt)]))
            print(labels)
            print(confusion_matrix(yt, yp, labels=labels))
