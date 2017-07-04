from keras.layers import LSTM, Activation, Dropout, Dense
from keras.models import Sequential
import numpy as np
import pandas as pd
from keras.regularizers import l2
import re
from os import listdir
from os.path import join, isfile
from email import parser as ep
from keras.utils import np_utils
from logger import log, is_lower

parser = ep.Parser()

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


def parse_xfrom(xfrom):
    fn = ln = mail = None

    match = re.search(r"(?P<ln>\w+), (?P<fn>\w+)(?: ?(?P<mn>\w)\.?)?", xfrom)
    if match:
        fn = match.group('fn')
        ln = match.group('ln')
    else:
        match = re.search(r"(?P<fn>\w+) (?:(?P<mn>\w)\.? )?(?P<ln>\w+)", xfrom)
        if match:
            fn = match.group('fn')
            ln = match.group('ln')
        else:
            fn = re.sub(r"\W", "", xfrom.split(' ')[0], flags=re.I)

    match = re.search(r"(?P<mail>[a-z0-9\.\-_]+@[a-z0-9\.\-_]+\.[a-z]{2,3})", xfrom, flags=re.I)
    if match:
        mail = match.group('mail')

    return fn, ln, mail


def boo(cond):
    return 1 if bool(cond) else -1


def line_features(body, li, mail):
    sb = body.splitlines()
    line = sb[li]
    cline = re.sub(r"^\s*(H|B|S)>(\s*>?)+", "", line)

    punctuation = r"(\!|\"|\#|\$|\%|\&|\\|\'|\(|\)|\*|\+|\,|\-|\.|\/|\:|\;|\<|\=|\>|\?|\@|\[|\]|\^|\_|\`|\{|\||\}|\~|\')"
    fn, ln, ml = parse_xfrom(mail['X-From'])
    l_line = len(line)

    feats = {
        'blank_line:0': boo(len(line.strip()) == 0),
        'emailpattern:0': boo(re.search(r"[a-z0-9\.\-_]+@[a-z0-9\.\-_]+\.[a-z]{2,3}", line, flags=re.I)),
        'lastline:0': boo((len(sb) - 1) == li),
        'prevToLastLine:0': boo((len(sb) - 2) == li),
        # email header pattern
        # url pattern
        # phone number pattern
        'signatureMarker:0': boo(re.match(r"^\s*\-\-\-*\s*$", line)),
        '5special:0': boo(re.search(r"^\s*(\*|#|\+|\^|\-|\~|_|\&|\/|\$|\!|\%|\:|\=){5,}", line)),
        # typical signature words (dept, university, corp,...)
        'namePattern:0': boo(re.search(r"[A-Z][a-z]+\s\s?[A-Z]\.?\s\s?[A-Z][a-z]+", line)),
        'quoteEnd:0': boo(re.search(r"\"$", line)),
        'containsSenderName_first:0': boo(bool(fn) and fn.lower() in line.lower()),
        'containsSenderName_last:0': boo(bool(ln) and ln.lower() in line.lower()),
        'containsSenderName_mail:0': boo(bool(ml) and ml.lower() in line.lower()),
        'numTabs=1:0': boo(line.count('\t') == 1),
        'numTabs=2:0': boo(line.count('\t') == 2),
        'numTabs>=3:0': boo(line.count('\t') >= 3),
        'punctuation>20:0': boo(l_line > 0 and (len(re.findall(punctuation, line)) / l_line) > 0.2),
        'punctuation>50:0': boo(l_line > 0 and (len(re.findall(punctuation, line)) / l_line) > 0.5),
        'punctuation>90:0': boo(l_line > 0 and (len(re.findall(punctuation, line)) / l_line) > 0.9),
        'typicalMarker:0': boo(re.search(r"^\>", line)),
        'startWithPunct:0': boo(re.search(r"^" + punctuation, line)),
        'nextSamePunct:0': boo(True if (li + 1 >= len(sb) or 0 == l_line == len(sb[li + 1])) else (
            bool(re.search(r"^" + punctuation, sb[li + 1])) and l_line > 0 and len(sb[li + 1]) > 0 and
            sb[li + 1][
                0] == line[0])),
        'prevSamePunct:0': boo(True if (li - 1 >= 0 or 0 == l_line == len(sb[li - 1])) else (
            bool(re.search(r"^" + punctuation, sb[li - 1])) and l_line > 0 and len(sb[li - 1]) > 0 and
            sb[li - 1][
                0] == line[0])),
        # starts with 1-2 punct followed by reply marker: "^\p{Punct}{1,2}\>"
        # reply line clue: "wrote:$" or "writes:$"
        'alphnum<90:0': boo(l_line > 0 and (len(re.findall('[a-zA-Z0-9]', line)) / l_line) < 0.9),
        'alphnum<50:0': boo(l_line > 0 and (len(re.findall('[a-zA-Z0-9]', line)) / l_line) < 0.5),
        'alphnum<10:0': boo(l_line > 0 and (len(re.findall('[a-zA-Z0-9]', line)) / l_line) < 0.1),
        'hasWord=fwdby:0': boo(re.search(r"forwarded by", line, flags=re.I)),
        'hasWord=origmsg:0': boo(re.search(r"original message", line, flags=re.I)),
        'hasWord=fwdmsg:0': boo(re.search(r"forwarded message", line, flags=re.I)),
        'hasWord=from:0': boo(re.search(r"from:", cline, flags=re.I)),
        'hasWord=to:0': boo(re.search(r"to:", cline, flags=re.I)),
        'hasWord=subject:0': boo(re.search(r"subject:", cline, flags=re.I)),
        'hasWord=cc:0': boo(re.search(r"cc:", cline, flags=re.I)),
        'hasWord=bcc:0': boo(re.search(r"bcc:", cline, flags=re.I)),
        'hasWord=subj:0': boo(re.search(r"subj:", cline, flags=re.I)),
        'hasWord=date:0': boo(re.search(r"date:", cline, flags=re.I)),
        'hasWord=sent:0': boo(re.search(r"sent:", cline, flags=re.I)),
        'hasWord=sentby:0': boo(re.search(r"sent by:", cline, flags=re.I)),
        'hasWord=fax:0': boo(re.search(r"fax", cline, flags=re.I)),
        'hasWord=phone:0': boo(re.search(r"phone", cline, flags=re.I)),
        'hasWord=cell:0': boo(re.search(r"phone", cline, flags=re.I)),
        'beginswithShape=Xx{2,8}\::0': boo(re.search(r"[A-Z][a-z]{1,7}:", cline)),
        'hasForm=^dd/dd/dddd dd:dd ww$:0': boo(
            re.search(r"^\s*\d\d\/\d\d\/\d\d\d\d \d?\d\:\d\d(\:\d\d)? ?(am|pm)?\s*$", cline, flags=re.I)),
        'hasForm=^dd:dd:dd ww$:0': boo(re.search(r"^\s*\d\d\:\d\d(\:\d\d)? ?(am|pm)\s*$", cline, flags=re.I)),
        'containsForm=dd/dd/dddd dd:dd ww:0': boo(
            re.search(r"on\s*\d\d\/\d\d\/\d\d\d\d \d?\d\:\d\d(\:\d\d)? ?(am|pm)?", cline, flags=re.I)),
        'hasLDAPthings': boo(re.search(r"\w+ \w+\/[A-Z]{1,4}\/[A-Z]{2,8}@[A-Z]{2,8}", cline))

    }
    feats['containsSenderName_any:0'] = boo(feats['containsSenderName_first:0'] > 0 or
                                            feats['containsSenderName_last:0'] > 0 or
                                            feats['containsSenderName_mail:0'] > 0)

    feats['containsMimeWord:0'] = boo(
        feats['hasWord=from:0'] > 0 or feats['hasWord=to:0'] > 0 or feats['hasWord=cc:0'] > 0 or
        feats['hasWord=bcc:0'] > 0 or feats['hasWord=subject:0'] > 0 or feats['hasWord=subj:0'] > 0 or
        feats['hasWord=date:0'] > 0 or feats['hasWord=sent:0'] > 0 or feats['hasWord=sentby:0'] > 0)

    feats['containsHeaderStart:0'] = boo(feats['hasWord=fwdby:0'] > 0 or feats['hasWord=origmsg:0'] > 0 or
                                         feats['hasWord=fwdmsg:0'] > 0)

    feats['containsSignatureWord:0'] = boo(
        feats['hasWord=fax:0'] > 0 or feats['hasWord=cell:0'] > 0 or feats['hasWord=phone:0'] > 0)

    return feats


class Splitter:
    def __init__(self, path_annotated, window_size, include_signature=False, features=None, training_epochs=10,
                 nb_slack_lines=4, retrain=True, model_path=None):
        log('INFO', 'Created Splitter instance. retrain=%s, ws=%d', retrain, window_size)
        self.window_size = window_size
        self.features = features if features is not None else cols
        self.path_train = path_annotated
        self.training_epochs = training_epochs
        self.nb_slack_lines = nb_slack_lines
        self.retrain = retrain
        self.model_path = model_path

        self._is_prepared = False

        if include_signature:
            log('INFO', 'Include signatures!')
            self.annotation_map = {'H': 0, 'B': 1, 'S': 2}
            self.annotation_map_inv = {0: 'H', 1: 'B', 2: 'S'}
        else:
            log('INFO', 'Not including signatures!')
            self.annotation_map = {'H': 0, 'B': 1, 'S': 1}
            self.annotation_map_inv = {0: 'H', 1: 'B'}

        self.nb_classes = len(set(self.annotation_map.values()))

    def prepare(self):
        try:
            if self.retrain:
                self._prepare_train()
            else:
                self._prepare_load_trained()
        except OSError as e:
            log('ERROR', 'Problem loading the specified weights file, falling back to training again...')
            log('ERROR', str(e))
            self._prepare_load_trained()
        self._is_prepared = True

    @property
    def is_prepared(self):
        return self._is_prepared

    def transform(self, mail, processed):
        body = mail.get_payload()
        tmpx = []
        spl = body.splitlines()
        for j in range(len(spl)):
            f = line_features(body, j, mail)
            f['mail'] = 0  # for compatibility
            f['anno'] = 'B'  # for compatibility
            f['anno_num'] = 1
            f['line'] = j
            f['text'] = spl[j]
            tmpx.append(f)
        frame = pd.DataFrame(tmpx)
        nested_X, _, _, nested_index = self._mail2nested(frame)

        nested_yp = self.model.predict_proba(np.array(nested_X), verbose=0)
        y_proba, y, index = self._flatten_prediction(nested_yp, nested_index)
        y_fixed, parts = self._prediction2split(y_proba, y, frame, index)

        parts = []
        tmppart = {'H': [], 'B': [], 'S': []}
        lasty = y_fixed[0]
        for y, line in zip(y_fixed, spl):
            if y == self.annotation_map['H'] and lasty != y:
                parts.append(('\n'.join(tmppart['H']), '\n'.join(tmppart['B']), '\n'.join(tmppart['S'])))
                tmppart = {'H': [], 'B': [], 'S': []}
            tmppart[self.annotation_map_inv[y]].append(line)
            lasty = y
        parts.append(('\n'.join(tmppart['H']), '\n'.join(tmppart['B']), '\n'.join(tmppart['S'])))

        return parts

    def _prepare_load_trained(self):
        _, self.model = self.init_lstm_net()
        log('INFO', 'Loading weights from file %s', self.model_path)
        self.model.load_weights(self.model_path, by_name=False)

    def _prepare_train(self):
        log('INFO', '[training] Reading files from directory: %s', self.path_train)
        mails, bodies, annotations = self._read_train_dir()

        log('INFO', '[training] Loaded %d email files, total lines: %d, average num of lines per mail: %.2f',
            len(mails), np.sum([len(a) for a in annotations]), np.mean([len(a) for a in annotations]))

        log('INFO', '[training] Adding features to raw data...')
        tmpx = []
        for i in range(len(mails)):
            spl = bodies[i].splitlines()
            for j in range(len(spl)):
                f = line_features(bodies[i], j, mails[i])
                f['mail'] = i
                f['line'] = j
                f['text'] = spl[j]
                f['anno'] = annotations[i][j]
                f['anno_num'] = self.annotation_map[annotations[i][j]]
                tmpx.append(f)

        X, y, Y, index = self._frame2nested(pd.DataFrame(tmpx))

        self.model_init_weights, self.model = self.init_lstm_net()

        v = 1 if is_lower('DEBUG', le=True) else 0
        self.model.fit(X, Y, batch_size=self.window_size, nb_epoch=self.training_epochs, verbose=v)

        if self.model_path:
            log('INFO', 'Saving weights to %s', self.model_path)
            self.model.save_weights(self.model_path, overwrite=True)
        else:
            log('INFO', 'No path given, not saving the weights.')

    def init_lstm_net(self):
        log('DEBUG', 'init LSTM')
        model = Sequential()
        model.add(LSTM(len(self.features),
                       batch_input_shape=(None, self.window_size, len(self.features)),
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

        model.add(Dense(self.nb_classes,
                        init='he_uniform',
                        activation='softmax',
                        name='output'))

        model.compile(loss='categorical_crossentropy',  # 'msle',
                      optimizer='RMSprop',  # opts.Adadelta(),
                      metrics=['accuracy'])

        if is_lower('INFO', le=True):
            model.summary()

        return model.get_weights(), model

    @staticmethod
    def _clean_train_body(s):
        # remove annotation
        s = re.sub("^(H|S|B)>", "", s, flags=re.M)

        # remove known common rubbish
        s = s.replace('=\n', '').replace('=20', '').replace('=09', '').replace('=01\&', '') \
            .replace('=01&', '').replace('=18', '').replace('=018', '')

        # remove indentation
        # s = re.sub(r"^(\s*>)+","", s)

        # remove attachments
        s = re.sub(r"\s*\[IMAGE\]\s*", "", s, flags=re.I)
        s = re.sub(r"<<.{3,50}\.(xls|xlsx|png|gif|jpg|jpeg|doc|docx|ppt|pptx|pst)>>%?", "", s, flags=re.I)
        s = re.sub(r"^\s*-.{3,50}\.(xls|xlsx|png|gif|jpg|jpeg|doc|docx|ppt|pptx|pst)%?", "", s, flags=re.I)
        return s

    def _read_train_dir(self, num=None, skip=0):
        mails = []
        bodies = []
        annos = []
        for i, f in enumerate(listdir(self.path_train)):
            if i < skip:
                continue
            if num and len(mails) >= num:
                log('INFO', 'Stop reading files, was limited to %d', num)
                break

            if isfile(join(self.path_train, f)):
                with open(join(self.path_train, f)) as file:
                    mail = parser.parsestr(file.read())
                    mails.append(mail)
                    annos.append([l[0] if len(l) > 0 else 'B' for l in iter(mail.get_payload().splitlines())])
                    bodies.append(Splitter._clean_train_body(mail.get_payload()))

        return mails, bodies, annos

    def _window2vec(self, window):
        # map boolean values to -1 and 1
        d = {True: 1, False: -1}
        # get X as matrix, first map boolean values to int
        # X = window.applymap(lambda x: d.get(x, x)).as_matrix(columns=self.features)
        # X = window.as_matrix(columns=self.features)
        X = window[self.features]._data.blocks[0].values.T

        # get Y
        tmpy = list(window['anno_num'])
        Y = np_utils.to_categorical(tmpy, nb_classes=self.nb_classes)
        y = tmpy

        index = list(window.index)

        return X, y, Y, index

    def _mail2nested(self, mailfrm):
        X, y, Y, index = [], [], [], []

        lines = mailfrm.sort_values(by='line')
        n_lines = len(lines)
        ws = (self.window_size if self.window_size else n_lines - 1)
        # log('TRACE', 'Mail %d of size %d at windowsize %d (%d)', m, len(body), ws, self.window_size)
        for wi in range(n_lines - ws + 1):
            # make frame of windowsize
            tmpX, tmpy, tmpY, tmpI = self._window2vec(lines[wi:(wi + ws)])
            X.append(tmpX)
            y.append(tmpy)
            Y.append(tmpY)
            index.append(tmpI)

        if n_lines < ws:
            tmpX, tmpy, tmpY, tmpI = self._window2vec(lines)
            tmpX = list(tmpX)
            tmpy = list(tmpy)
            tmpY = list(tmpY)
            for i in range(n_lines, ws):
                tmpX.append(tmpX[-1])
                tmpy.append(tmpy[-1])
                tmpY.append(tmpY[-1])
                tmpI.append(tmpI[-1])
            X = [tmpX]
            y = [tmpy]
            Y = [tmpY]
            index = [tmpI]

        return X, y, Y, index

    def _frame2nested(self, frm):
        retx = []
        rety = []
        retY = []
        reti = []
        for m, body in frm.groupby('mail'):
            X, y, Y, index = self._mail2nested(body)
            retx += X
            rety += y
            retY += Y
            reti += index

        return np.array(retx), np.array(rety), np.array(retY), reti

    def _flatten_prediction(self, y_pred_proba_nested, nested_index):
        predictions = {}
        for i, (window, yp) in enumerate(zip(nested_index, y_pred_proba_nested)):
            for j, mail_i in enumerate(window):
                if mail_i not in predictions:
                    predictions[mail_i] = []
                predictions[mail_i].append(y_pred_proba_nested[i][j])

        flat_index = list(predictions.keys())
        y_pred_proba_flat = []
        y_pred_flat = []
        for li, n in enumerate(flat_index):
            tmp = np.nanmean(predictions[n], axis=0)
            y_pred_flat.append(np.nanargmax(tmp))
            y_pred_proba_flat.append(tmp)
        return np.array(y_pred_proba_flat), np.array(y_pred_flat), flat_index

    def _getnxtchange(self, annos, i, fwd=True):
        """
        :param i: current position
        :param fwd: True if towards inc i, False if dec i
        :param max_steps: number of steps to take in this direction
        :return: 0 if immediate change, no change within limit==limit
        """
        di = 1 if fwd else -1
        steps_taken = 0
        for ri in range(i + di, i + (di * self.nb_slack_lines), di):
            if ri < 0 or ri >= len(annos):
                return self.nb_slack_lines
            elif annos[i] != annos[ri]:
                return abs(i - ri) - 1
            else:
                steps_taken += 1

        return steps_taken

    def _prediction2split(self, y_pred_proba, y_pred, frame, index):
        parts = []
        yp_fixed = []

        annot = [{0: 'H', 1: 'B', 2: 'S'}[pfi] for pfi in y_pred]

        log('MICROTRACE', "--------START-MAIL----------")

        lastsym = annot[0]
        breaks = []
        for li, ann in enumerate(annot):
            if lastsym != ann and self._getnxtchange(annot, li) > 1:
                lastsym = ann
                breaks.append(li)

        typ = {'H': 0, 'B': 0, 'S': 0}
        tmpmail = {}
        partcount = 0
        tmpblob = []
        for li, (n, line) in enumerate(frame.loc[index].iterrows()):
            try:
                if li in breaks:
                    log('MICROTRACE', " ## BREAK ## %s -> %s", typ,
                        list(np.array(list(typ.values())) / np.array(list(typ.values())).sum()))

                    # find the dominating annotation type
                    blobtype = max(typ, key=typ.get)

                    # join the blob into text and add to email
                    tmpmail[blobtype] = "\n".join(tmpblob)

                    # add adjusted annotation
                    yp_fixed += [self.annotation_map[blobtype]] * len(tmpblob)
                    log('MICROTRACE', 'Adding to yp_fixed += %s', [self.annotation_map[blobtype]] * len(tmpblob))

                    # clear helpers
                    tmpblob = []
                    typ = {'H': 0, 'B': 0, 'S': 0}

                    # if this email part (header + body) is complete
                    if annot[li - 1 if li > 0 else 0] == 'B':
                        tmpmail['partcount'] = partcount
                        parts.append(tmpmail)
                        tmpmail = {}
                        partcount += 1
                        log('MICROTRACE', ' ### MAILBREAK ###')

                tmpblob.append(line['text'])

                log('MICROTRACE', '%s %3d. (%s) %s> %s',
                    line.get('anno', ''),
                    li,
                    (", ".join(["%.3f" % p for p in y_pred_proba[li]])),
                    {'H': 'H--', 'B': '-B-', 'S': '--S'}[annot[li]] or '---',
                    line['text'])
                typ[annot[li]] += 1

            except KeyError:
                log('MICROTRACE', "     (X.XXX, X.XXX) ---> %s", line['text'])
                pass

        blobtype = max(typ, key=typ.get)
        tmpmail[blobtype] = "\n".join(tmpblob)
        yp_fixed += [self.annotation_map[blobtype]] * len(tmpblob)
        tmpmail['partcount'] = partcount
        parts.append(tmpmail)

        log('MICROTRACE', "---------END-MAIL-----------")

        return yp_fixed, parts
