from keras.layers import LSTM, Activation, Dropout, Dense
from keras.models import Sequential
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from keras.utils import np_utils
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from optparse import OptionParser
from split import EmailHolder
import logging
import sys

oparser = OptionParser()
oparser.add_option("-l", "--log-level",
                   dest="log_level",
                   metavar="LEVEL",
                   help="set log level to LEVEL",
                   type='choice',
                   choices=['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'],
                   default='INFO')
oparser.add_option("-a", "--path-annotated",
                   dest="path_annotated",
                   metavar="DIR",
                   help="recursively read annotated email files from DIR",
                   default='/home/tim/Uni/HPI/workspace/enron_data/splitting/annotated/')
oparser.add_option("-m", "--path-more",
                   dest="path_more",
                   metavar="DIR",
                   help="recursively read raw email files from DIR",
                   default='/home/tim/Uni/HPI/workspace/enron_data/splitting/data/')
oparser.add_option("-t", "--train-size",
                   dest="splitratio",
                   metavar="RATIO",
                   help="pick float between 0-1 as the portion of annotated files used as training samples",
                   type='float',
                   default=0.7)
oparser.add_option("-w", "--window-size",
                   dest="windowsize",
                   metavar="SIZE",
                   help="number of lines to consider as window size",
                   type='int',
                   default=8)
oparser.add_option("-n", "--num-files",
                   dest="num_files",
                   metavar="NUM",
                   help="number of files (NUM) to read",
                   type='int',
                   default=None)
oparser.add_option("-e", "--epochs",
                   dest="training_epochs",
                   metavar="NUM",
                   help="number of epochs (NUM) to train for",
                   type='int',
                   default=10)
oparser.add_option("-s", "--include-signatures",
                   dest="include_signature",
                   help="set this flag to include signature detection",
                   action="store_true")

(options, args) = oparser.parse_args()

logging.addLevelName(5, 'TRACE')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')

# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# logging.root.addHandler(ch)
hdlr = logging.FileHandler('log')
hdlr.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
logging.root.addHandler(hdlr)


def log(lvl, msg, *args, **kwargs):
    logging.log(logging.getLevelName(lvl), msg, *args, **kwargs)


cols = ['5special:0',
        'alphnum<10:0',
        'alphnum<50:0',
        'alphnum<90:0',
        # 'anno',
        'blank_line:0',
        'containsHeaderStart:0',
        'containsMimeWord:0',
        'containsSenderName_any:0',
        'containsSenderName_first:0',
        'containsSenderName_last:0',
        'containsSenderName_mail:0',
        'containsSignatureWord:0',
        'emailpattern:0',
        # 'hasWord=bcc:0',
        # 'hasWord=cc:0',
        # 'hasWord=cell:0',
        # 'hasWord=date:0',
        # 'hasWord=fax:0',
        # 'hasWord=from:0',
        # 'hasWord=fwdby:0',
        # 'hasWord=fwdmsg:0',
        # 'hasWord=origmsg:0',
        # 'hasWord=phone:0',
        # 'hasWord=sent:0',
        # 'hasWord=subj:0',
        # 'hasWord=subject:0',
        # 'hasWord=to:0',
        'lastline:0',
        # 'line',
        # 'mail',
        'namePattern:0',
        'nextSamePunct:0',
        'numTabs=1:0',
        'numTabs=2:0',
        'numTabs>=3:0',
        'prevSamePunct:0',
        'prevToLastLine:0',
        'punctuation>20:0',
        'punctuation>50:0',
        'punctuation>90:0',
        'quoteEnd:0',
        'signatureMarker:0',
        'startWithPunct:0',
        # 'text',
        'typicalMarker:0',
        # 'beginswithShape=Xx{2,8}\::0',
        # 'hasForm=^dd/dd/dddd dd:dd ww$:0',
        # 'containsForm=dd/dd/dddd dd:dd ww:0',
        # 'hasLDAPthings:0',
        # 'hasForm=^dd:dd:dd ww$:0'
        ]


class SplitEval:
    def __init__(self, path_annotated, path_more, window_size, include_signature=False,
                 features=None, num_files=None, train_size=0.7):
        log('INFO', 'Init split eval...')
        self.window_size = window_size
        self.features = features if features is not None else cols
        self.path_more = path_more

        if include_signature:
            log('INFO', 'Include signatures!')
            self.annotation_map = {'H': 0, 'B': 1, 'S': 2}
            self.annotation_map_inv = {0: 'H', 1: 'B', 2: 'S'}
        else:
            log('INFO', 'Not including signatures!')
            self.annotation_map = {'H': 0, 'B': 1, 'S': 1}
            self.annotation_map_inv = {0: 'H', 1: 'B'}

        log('INFO', 'Loading emails...')
        self.mails = EmailHolder(path_annotated, window_size, self.annotation_map, self.annotation_map_inv,
                                 features, train_size, num_files)

        self.X_train, self.y_train, self.Y_train, self.I_train = self.mails.train_nested
        self.X_test, self.y_test, self.Y_test, self.I_test = self.mails.test_nested

    def init_LSTMnet(self):
        model = Sequential()
        model.add(LSTM(self.mails.get_num_features(),
                       batch_input_shape=(None, self.window_size, self.mails.get_num_features()),
                       return_sequences=True,
                       init='he_uniform',
                       inner_init='orthogonal',
                       W_regularizer=l2(0.01),
                       U_regularizer=l2(0.01),
                       name='rnn_layer1'))
        model.add(Activation('softsign'))
        model.add(Dropout(0.2))
        model.add(LSTM(50,
                       return_sequences=True,
                       init='he_uniform',
                       inner_init='orthogonal',
                       W_regularizer=l2(0.01),
                       U_regularizer=l2(0.01),
                       name='rnn_layer2'))
        model.add(Activation('softsign'))
        model.add(Dropout(0.4))

        model.add(Dense(self.mails.get_num_targets(),
                        init='he_uniform',
                        activation='softmax',
                        name='output'))

        model.compile(loss='categorical_crossentropy',  # 'msle',
                      optimizer='RMSprop',  # opts.Adadelta(),
                      metrics=['accuracy'])

        if logging.getLogger().getEffectiveLevel() < logging.INFO:
            model.summary()

        return model.get_weights(), model

    def reset_model(self, model, weights=None):
        if weights is None:
            weights = model.get_weights()
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        model.set_weights(weights)

    def fit_model(self, model, epochs, verbose=None):
        v = logging.getLogger().getEffectiveLevel() <= logging.DEBUG if verbose is None else verbose
        return model.fit(self.X_train, self.Y_train, batch_size=self.window_size, nb_epoch=epochs, verbose=v,
                         validation_data=(self.X_test, self.Y_test)).history

    def fit_model_continue(self, model, epochs, hist, verbose=None):
        newhist = self.fit_model(model, epochs, verbose).history
        return {k: v + newhist[k] for k, v in hist.items()}

    def predict(self, model, mailholder, verbose=None, defer_fix=True):
        v = logging.getLogger().getEffectiveLevel() <= logging.DEBUG if verbose is None else verbose
        y_pred = model.predict_proba(mailholder.get_features_nested(), verbose=v)
        return NestedResultHolder(mailholder, y_pred, defer_split=defer_fix)

    def evaluate(self, holder, include_fixed=False):
        """
        :param NestedResultHolder holder:
        :return:
        """
        cnt, corr, cntdeep, corrdeep = holder.count_correct()

        print('correct windows:', corr, "/", cnt, "=", (corr / cnt))
        print('correct lines', corrdeep, "/", cntdeep, "=", (corrdeep / cntdeep))
        print("================================")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("==== each line in all the windows ====")
            print(classification_report([k for u in holder.y_nested for k in u],
                                        [k for u in holder.y_pred_nested for k in u],
                                        target_names=['(H)eader', '(B)ody', '(S)ignature']))
            print(confusion_matrix([k for u in holder.y_nested for k in u],
                                   [k for u in holder.y_pred_nested for k in u]))

            print("\n==== each line based on merged predictions ====")
            print(classification_report(holder.y_flat, holder.y_pred_flat,
                                        target_names=['(H)eader', '(B)ody', '(S)ignature']))
            print(confusion_matrix(holder.y_flat, holder.y_pred_flat))

            if include_fixed:
                print("\n==== each line based on fixed predictions ====")
                print(classification_report(holder.y_flat, holder.y_pred_flat_fixed,
                                            target_names=['(H)eader', '(B)ody', '(S)ignature']))
                print(confusion_matrix(holder.y_flat, holder.y_pred_flat_fixed))

    def draw_traingraph(self, hist):
        plt.subplot(121)
        plt.plot(hist['acc'], label='accuracy')
        if 'val_acc' in hist:
            plt.plot(hist['val_acc'], label='eval_accuracy')
        plt.legend(loc='lower right')

        plt.subplot(122)
        plt.plot(hist['loss'], label='loss')
        if 'val_loss' in hist:
            plt.plot(hist['val_loss'], label='eval_loss')
        plt.legend(loc='upper right')
        plt.show()

    def predict_unseen(self, skip, lim=20):
        splitmails = []

        # TODO
        # holder = EmailHolder(path_annotated, window_size, self.annotation_map, self.annotation_map_inv,
        #            features, 0, num_files)


class NestedResultHolder:
    def __init__(self, mailholder, Yp, defer_split=True):
        """
        :param EmailHolder mailholder: email holder used for prediction
        """
        self.mailholder = mailholder

        self.annotation_map_inv = mailholder.annotation_map_inv
        self.annotation_map = mailholder.annotation_map
        self.nested_index = mailholder.get_nested_index(train=False)
        self.flat_index = mailholder.get_flat_index(train=False)

        self.y_flat = mailholder.get_labels_flat(train=False, onehot=False)
        self.y_nested = mailholder.get_labels_nested(train=False, onehot=False)
        self.y_flat_s = [self.annotation_map_inv[yi] for yi in self.y_flat]

        self.y_pred_proba_nested = Yp

        log('INFO', '   Merging predictions...')
        self.predictions, self.y_pred_nested = self._match_predictions()
        self.y_pred_flat, self.y_pred_proba_flat = self._flatten_prediction()

        if not defer_split:
            log('INFO', 'Running heuristics to split mails...')
            self.y_pred_flat_fixed = self.splitmails()

    def _match_predictions(self):
        preds = {}
        cat = []
        for i, (window, yp) in enumerate(zip(self.nested_index, self.y_pred_proba_nested)):
            cat.append(np.nanargmax(yp, axis=1))
            for j, mail_i in enumerate(window):
                if mail_i not in preds:
                    preds[mail_i] = []
                preds[mail_i].append(self.y_pred_proba_nested[i][j])
        return preds, np.array(cat)

    def _flatten_prediction(self):
        # print(self.predictions)
        # print(self.flat_index)
        # print(self.nested_index)

        a = []
        for m in self.nested_index:
            for n in m:
                a.append(n)

        predflat = []
        predflat_p = []
        for li, n in enumerate(self.flat_index):
            tmp = np.nanmean(self.predictions[n], axis=0)
            tmpa = np.nanargmax(tmp)
            predflat.append(tmpa)
            predflat_p.append(tmp)
        return np.array(predflat), np.array(predflat_p)

    def count_correct(self):
        t = self.y_nested
        p = self.y_pred_nested
        log('DEBUG', 'count correct() - y_nested.shape: %s, y_pred_nested.shape: %s', t.shape, p.shape)
        cnt = 0
        corr = 0
        cntdeep = 0
        corrdeep = 0
        for i, a in enumerate(t):
            tmp = True
            for j, b in enumerate(a):
                tmp = tmp and (t[i][j] == p[i][j])
                corrdeep += 1 if (t[i][j] == p[i][j]) else 0
                cntdeep += 1
            if tmp:
                corr += 1
            cnt += 1
        return cnt, corr, cntdeep, corrdeep

    def _getnxtchange(self, annos, i, fwd=True, max_steps=8):
        """
        :param i: current position
        :param fwd: True if towards inc i, False if dec i
        :param max_steps: number of steps to take in this direction
        :return: 0 if immediate change, no change within limit==limit
        """
        di = 1 if fwd else -1
        steps_taken = 0
        for ri in range(i + di, i + (di * max_steps), di):
            if ri < 0 or ri >= len(annos):
                return max_steps
            elif annos[i] != annos[ri]:
                return abs(i - ri) - 1
            else:
                steps_taken += 1

        return steps_taken

    def splitmails(self, **kwargs):
        tmp = self.mailholder.frame.loc[self.flat_index].copy()
        anp = pd.DataFrame(self.y_pred_proba_flat, index=tmp.index)
        an = pd.Series(self.y_pred_flat, index=tmp.index)

        mails = []
        y_fixed = []
        for mi, mail in tmp.groupby('mail'):
            parts, yp_fixed = self.splitmail(mi, mail, np.array(an.loc[mail.index]), anp.loc[mail.index].as_matrix(),
                                             **kwargs)
            mails.append(parts)
            y_fixed.append(yp_fixed)
        return mails, np.array([item for sublist in y_fixed for item in sublist])

    def splitmail(self, mail_id, subframe, y_pred_flat, y_pred_proba_flat, max_steps=4):

        parts = []
        yp_fixed = []

        annot = [{0: 'H', 1: 'B', 2: 'S'}[pfi] for pfi in y_pred_flat]

        log('TRACE', "--------START-MAIL----------")

        lastchange = 0
        lastsym = annot[0]
        breaks = []
        for li, ann in enumerate(annot):
            if lastsym != ann and self._getnxtchange(annot, li, max_steps=max_steps) > 1:
                lastsym = ann
                lastchange = li
                breaks.append(li)

        typ = {'H': 0, 'B': 0, 'S': 0}
        tmpmail = {}
        partcount = 0
        tmpblob = []
        for li, (n, line) in enumerate(subframe.iterrows()):
            try:
                typ[annot[li]] += 1
                if li in breaks:
                    log('TRACE', " ## BREAK ## %d -> %f", typ,
                        np.array(list(typ.values())) / np.array(list(typ.values())).sum())

                    # find the dominating annotation type
                    blobtype = max(typ, key=typ.get)

                    # join the blob into text and add to email
                    tmpmail[blobtype] = "\n".join(tmpblob)

                    # add ajusted annotation
                    yp_fixed += [self.annotation_map[blobtype]] * len(tmpblob)

                    # clear helpers
                    tmpblob = []
                    typ = {'H': 0, 'B': 0, 'S': 0}

                    # if this email part (header + body) is complete
                    if annot[li - 1 if li > 0 else 0] == 'B':
                        tmpmail['partcount'] = partcount
                        tmpmail['mail'] = mail_id
                        parts.append(tmpmail)
                        tmpmail = {}
                        partcount += 1
                        log('TRACE', ' ### MAILBREAK ###')

                tmpblob.append(line['text'])

                log('TRACE', '%s %3d. (%s) %s> %s',
                    line.get('anno', ''),
                    li,
                    (", ".join(["%.3f" % p for p in y_pred_proba_flat[li]])),
                    {'H': 'H--', 'B': '-B-', 'S': '--S'}[annot[li]] or '---',
                    line['text'])

            except KeyError:
                log('TRACE', "     (X.XXX, X.XXX) ---> %s", line['text'])
                pass

        blobtype = max(typ, key=typ.get)
        tmpmail[blobtype] = "\n".join(tmpblob)
        yp_fixed += [self.annotation_map[blobtype]] * len(tmpblob)
        tmpmail['partcount'] = partcount
        tmpmail['mail'] = mail_id
        parts.append(tmpmail)

        log('TRACE', "---------END-MAIL-----------")

        return parts, yp_fixed


if __name__ == "__main__":
    logging.root.setLevel(logging.getLevelName(options.log_level))

    evaluator = SplitEval(options.path_annotated, options.path_more, options.windowsize, options.include_signature,
                          features=cols, num_files=options.num_files, train_size=options.splitratio)

    log('INFO', 'Init net...')
    model_weights, model = evaluator.init_LSTMnet()
    log('INFO', 'training net...')
    evaluator.fit_model(model, options.training_epochs, verbose=None)
    log('INFO', 'getting predictions...')
    predictions = evaluator.predict(model, evaluator.mails, verbose=None, defer_fix=True)

    evaluator.evaluate(predictions, include_fixed=False)

    mails = EmailHolder(options.path_more, options.windowsize, evaluator.annotation_map, evaluator.annotation_map_inv,
                        cols, 0, 10)
    predictions_unseen = evaluator.predict(model, mails, verbose=None, defer_fix=False)
