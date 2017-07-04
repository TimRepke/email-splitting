from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import spacy
import re
from os import listdir
from os.path import isfile, join
import re
from email import parser as ep
from optparse import OptionParser
from keras.utils import np_utils
import logging

parser = ep.Parser()


def log(lvl, msg, *args, **kwargs):
    logging.log(logging.getLevelName(lvl), msg, *args, **kwargs)


class EmailHolder:
    def __init__(self, path, window_size, annotation_map, annotation_map_inv, features,
                 train_size=0.7, num_files=None, skip=0):
        self.window_size = window_size
        self.annotation_map = annotation_map
        self.annotation_map_inv = annotation_map_inv
        self.features = features

        log('INFO', 'Reading files from directory: %s', path)
        self.mails, bodies, annotations = self._read_dir(path, num=num_files, skip=skip)
        log('INFO', 'Loaded %d email files, total lines: %d, average num of lines per mail: %.2f',
            len(self.mails), np.sum([len(a) for a in annotations]), np.mean([len(a) for a in annotations]))

        log('INFO', 'Adding features to raw data...')
        x = []
        for i in range(len(self.mails)):
            spl = bodies[i].splitlines()
            for j in range(len(spl)):
                f = self._line_features(bodies[i], j, self.mails[i])
                f['mail'] = i
                f['line'] = j
                f['text'] = spl[j]
                f['anno'] = annotations[i][j]
                x.append(f)
        frame = pd.DataFrame(x)
        self.frame = frame
        self.I_flat = []

        log('INFO', 'Splitting train/test set')
        self.train_samples = np.random.choice(np.array(list(set(frame['mail']))),
                                              size=int(len(bodies) * train_size),
                                              replace=False)

        self.train = frame[frame['mail'].isin(list(self.train_samples))]
        self.test = frame[~frame['mail'].isin(list(self.train_samples))]

        log('INFO', 'Picked train/test set with a ratio %.2f/%.2f resulting in a total of %d/%d mails (%d/%d lines).',
            train_size, 1 - train_size,
            len(self.train_samples), len(set(frame['mail'])) - len(self.train_samples),
            len(self.train), len(self.test))

        self.train_nested = self._frame2nested(self.train)
        self.train_flat = self._frame2flat(self.train)

        self.test_nested = self._frame2nested(self.test)
        self.test_flat = self._frame2flat(self.test)

    def get_features_flat(self, train=False):
        if train:
            ret, _, _, _ = self.train_flat
        else:
            ret, _, _, _ = self.test_flat
        return ret

    def get_features_nested(self, train=False):
        if train:
            ret, _, _, _ = self.train_nested
        else:
            ret, _, _, _ = self.test_nested
        return ret

    def get_labels_flat(self, train=False, onehot=False):
        if train and onehot:
            _, _, ret, _ = self.train_flat
        elif train and not onehot:
            _, ret, _, _ = self.train_flat
        elif not train and onehot:
            _, _, ret, _ = self.test_flat
        else:
            _, ret, _, _ = self.test_flat

        return ret

    def get_labels_nested(self, train=False, onehot=False):
        if train and onehot:
            _, _, ret, _ = self.train_nested
        elif train and not onehot:
            _, ret, _, _ = self.train_nested
        elif not train and onehot:
            _, _, ret, _ = self.test_nested
        else:
            _, ret, _, _ = self.test_nested

        return ret

    def get_nested_index(self, train=False):
        if train:
            _, _, _, ret = self.train_nested
        else:
            _, _, _, ret = self.test_nested
        return ret

    def get_flat_index(self, train=False):
        if train:
            _, _, _, ret = self.train_flat
        else:
            _, _, _, ret = self.test_flat
        return ret

    def get_num_features(self):
        return len(self.features)

    def get_num_targets(self):
        return len(set(self.annotation_map.values()))

    def _clean_body(self, s):
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

    def _read_dir(self, mailpath, num=None, skip=0):
        mails = []
        bodies = []
        annos = []
        for i, f in enumerate(listdir(mailpath)):
            if i < skip:
                continue
            if num and len(mails) >= num:
                log('INFO', 'Stop reading files, was limited to %d', num)
                break

            if isfile(join(mailpath, f)):
                with open(join(mailpath, f)) as file:
                    mail = parser.parsestr(file.read())
                    mails.append(mail)
                    annos.append([l[0] if len(l) > 0 else 'B' for l in iter(mail.get_payload().splitlines())])
                    bodies.append(self._clean_body(mail.get_payload()))

        return mails, bodies, annos

    def _parse_xfrom(self, xfrom):
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

    def _line_features(self, body, li, mail):
        sb = body.splitlines()
        line = sb[li]
        cline = re.sub(r"^\s*(H|B|S)>(\s*>?)+", "", line)

        punctuation = r"(\!|\"|\#|\$|\%|\&|\\|\'|\(|\)|\*|\+|\,|\-|\.|\/|\:|\;|\<|\=|\>|\?|\@|\[|\]|\^|\_|\`|\{|\||\}|\~|\')"
        fn, ln, ml = self._parse_xfrom(mail['X-From'])

        feats = {
            'blank_line:0': len(line.strip()) == 0,
            'emailpattern:0': bool(re.search(r"[a-z0-9\.\-_]+@[a-z0-9\.\-_]+\.[a-z]{2,3}", line, flags=re.I)),
            'lastline:0': (len(sb) - 1) == li,
            'prevToLastLine:0': (len(sb) - 2) == li,
            # email header pattern
            # url pattern
            # phone number pattern
            'signatureMarker:0': bool(re.match(r"^\s*\-\-\-*\s*$", line)),
            '5special:0': bool(re.search(r"^\s*(\*|#|\+|\^|\-|\~|_|\&|\/|\$|\!|\%|\:|\=){5,}", line)),
            # typical signature words (dept, university, corp,...)
            'namePattern:0': bool(re.search(r"[A-Z][a-z]+\s\s?[A-Z]\.?\s\s?[A-Z][a-z]+", line)),
            'quoteEnd:0': bool(re.search(r"\"$", line)),
            'containsSenderName_first:0': bool(fn) and fn.lower() in line.lower(),
            'containsSenderName_last:0': bool(ln) and ln.lower() in line.lower(),
            'containsSenderName_mail:0': bool(ml) and ml.lower() in line.lower(),
            'numTabs=1:0': line.count('\t') == 1,
            'numTabs=2:0': line.count('\t') == 2,
            'numTabs>=3:0': line.count('\t') >= 3,
            'punctuation>20:0': len(line) > 0 and (len(re.findall(punctuation, line)) / len(line)) > 0.2,
            'punctuation>50:0': len(line) > 0 and (len(re.findall(punctuation, line)) / len(line)) > 0.5,
            'punctuation>90:0': len(line) > 0 and (len(re.findall(punctuation, line)) / len(line)) > 0.9,
            'typicalMarker:0': bool(re.search(r"^\>", line)),
            'startWithPunct:0': bool(re.search(r"^" + punctuation, line)),
            'nextSamePunct:0': True if (li + 1 >= len(sb) or 0 == len(line) == len(sb[li + 1])) else (
                bool(re.search(r"^" + punctuation, sb[li + 1])) and len(line) > 0 and len(sb[li + 1]) > 0 and
                sb[li + 1][
                    0] == line[0]),
            'prevSamePunct:0': True if (li - 1 >= 0 or 0 == len(line) == len(sb[li - 1])) else (
                bool(re.search(r"^" + punctuation, sb[li - 1])) and len(line) > 0 and len(sb[li - 1]) > 0 and
                sb[li - 1][
                    0] == line[0]),
            # starts with 1-2 punct followed by reply marker: "^\p{Punct}{1,2}\>"
            # reply line clue: "wrote:$" or "writes:$"
            'alphnum<90:0': len(line) > 0 and (len(re.findall('[a-zA-Z0-9]', line)) / len(line)) < 0.9,
            'alphnum<50:0': len(line) > 0 and (len(re.findall('[a-zA-Z0-9]', line)) / len(line)) < 0.5,
            'alphnum<10:0': len(line) > 0 and (len(re.findall('[a-zA-Z0-9]', line)) / len(line)) < 0.1,
            'hasWord=fwdby:0': bool(re.search(r"forwarded by", line, flags=re.I)),
            'hasWord=origmsg:0': bool(re.search(r"original message", line, flags=re.I)),
            'hasWord=fwdmsg:0': bool(re.search(r"forwarded message", line, flags=re.I)),
            'hasWord=from:0': bool(re.search(r"from:", cline, flags=re.I)),
            'hasWord=to:0': bool(re.search(r"to:", cline, flags=re.I)),
            'hasWord=subject:0': bool(re.search(r"subject:", cline, flags=re.I)),
            'hasWord=cc:0': bool(re.search(r"cc:", cline, flags=re.I)),
            'hasWord=bcc:0': bool(re.search(r"bcc:", cline, flags=re.I)),
            'hasWord=subj:0': bool(re.search(r"subj:", cline, flags=re.I)),
            'hasWord=date:0': bool(re.search(r"date:", cline, flags=re.I)),
            'hasWord=sent:0': bool(re.search(r"sent:", cline, flags=re.I)),
            'hasWord=sentby:0': bool(re.search(r"sent by:", cline, flags=re.I)),
            'hasWord=fax:0': bool(re.search(r"fax", cline, flags=re.I)),
            'hasWord=phone:0': bool(re.search(r"phone", cline, flags=re.I)),
            'hasWord=cell:0': bool(re.search(r"phone", cline, flags=re.I)),
            'beginswithShape=Xx{2,8}\::0': bool(re.search(r"[A-Z][a-z]{1,7}:", cline)),
            'hasForm=^dd/dd/dddd dd:dd ww$:0': bool(
                re.search(r"^\s*\d\d\/\d\d\/\d\d\d\d \d?\d\:\d\d(\:\d\d)? ?(am|pm)?\s*$", cline, flags=re.I)),
            'hasForm=^dd:dd:dd ww$:0': bool(re.search(r"^\s*\d\d\:\d\d(\:\d\d)? ?(am|pm)\s*$", cline, flags=re.I)),
            'containsForm=dd/dd/dddd dd:dd ww:0': bool(
                re.search(r"on\s*\d\d\/\d\d\/\d\d\d\d \d?\d\:\d\d(\:\d\d)? ?(am|pm)?", cline, flags=re.I)),
            'hasLDAPthings': bool(re.search(r"\w+ \w+\/[A-Z]{1,4}\/[A-Z]{2,8}@[A-Z]{2,8}", cline))

        }
        feats['containsSenderName_any:0'] = feats['containsSenderName_first:0'] or \
                                            feats['containsSenderName_last:0'] or \
                                            feats['containsSenderName_mail:0']

        feats['containsMimeWord:0'] = feats['hasWord=from:0'] or feats['hasWord=to:0'] or feats['hasWord=cc:0'] or \
                                      feats['hasWord=bcc:0'] or feats['hasWord=subject:0'] or feats['hasWord=subj:0'] or \
                                      feats['hasWord=date:0'] or feats['hasWord=sent:0'] or feats['hasWord=sentby:0']

        feats['containsHeaderStart:0'] = feats['hasWord=fwdby:0'] or feats['hasWord=origmsg:0'] or \
                                         feats['hasWord=fwdmsg:0']

        feats['containsSignatureWord:0'] = feats['hasWord=fax:0'] or feats['hasWord=cell:0'] or feats['hasWord=phone:0']

        return feats

    def _frame2nested(self, frm):
        retx = []
        rety = []
        retY = []
        reti = []
        reti_flat = []
        d = {True: 1, False: -1}
        for m, body in frm.groupby('mail'):
            lines = body.sort_values(by='line')
            ws = (self.window_size if self.window_size else len(lines) - 1)
            #log('TRACE', 'Mail %d of size %d at windowsize %d (%d)', m, len(body), ws, self.window_size)
            for wi in range(len(lines) - ws + 1):
                # make frame of windowsize
                window = lines[wi:(wi + ws)]
                #if m > 70: log('TRACE', 'wi: %d, wi+ws: %d, %s', wi, wi+ws, list(window.index))

                # get X
                retx.append(window.applymap(lambda x: d.get(x, x)).as_matrix(columns=self.features))

                # get Y
                tmpy = list(window['anno'].apply(lambda x: self.annotation_map.get(x.upper(), 0)))
                retY.append(np_utils.to_categorical(tmpy, nb_classes=len(set(self.annotation_map.values()))))
                rety.append(tmpy)

                reti.append(list(window.index))
                reti_flat.append(lines.iloc[wi].name)

            # add the rest of the last window
            if len(lines) >= ws:
                for wii in range(wi+1, len(lines)):
                    reti_flat.append(lines.iloc[wii].name)

        self.I_flat = reti_flat
        return np.array(retx), np.array(rety), np.array(retY), reti

    def _frame2flat(self, frm):
        frm = frm.loc[list(self.I_flat)].sort_values(by=['mail', 'line'])
        #frm = frm.sort_values(by=['mail', 'line'])
        X = frm.as_matrix(columns=self.features)
        y = frm['anno'].map(self.annotation_map).fillna(0.0)

        return X, np.array(y), \
               np_utils.to_categorical(y, nb_classes=len(set(self.annotation_map.values()))), list(frm.index)


class PresplitEmailHolder(EmailHolder):
    def __init__(self, path, window_size, annotation_map, annotation_map_inv, features,
                 train_size=0.7, num_files=None, skip=0):
        self.window_size = window_size
        self.annotation_map = annotation_map
        self.annotation_map_inv = annotation_map_inv
        self.features = features

        log('INFO', 'Reading files from directory: %s', path)
        self.mails, bodies, annotations = self._read_dir(path, num=num_files, skip=skip)
        log('INFO', 'Loaded %d email files, total lines: %d, average num of lines per mail: %.2f',
            len(self.mails), np.sum([len(a) for a in annotations]), np.mean([len(a) for a in annotations]))

        log('INFO', 'Adding features to raw data...')
        x = []
        for i in range(len(self.mails)):
            spl = bodies[i].splitlines()
            for j in range(len(spl)):
                f = self._line_features(bodies[i], j, self.mails[i])
                f['mail'] = i
                f['line'] = j
                f['text'] = spl[j]
                f['anno'] = annotations[i][j]
                x.append(f)
        frame = pd.DataFrame(x)
        self.frame = frame
        self.I_flat = []

        log('INFO', 'Splitting train/test set')
        self.train_samples = np.random.choice(np.array(list(set(frame['mail']))),
                                              size=int(len(bodies) * train_size),
                                              replace=False)

        self.train = frame[frame['mail'].isin(list(self.train_samples))]
        self.test = frame[~frame['mail'].isin(list(self.train_samples))]

        log('INFO', 'Picked train/test set with a ratio %.2f/%.2f resulting in a total of %d/%d mails (%d/%d lines).',
            train_size, 1 - train_size,
            len(self.train_samples), len(set(frame['mail'])) - len(self.train_samples),
            len(self.train), len(self.test))

        self.train_nested = self._frame2nested(self.train)
        self.train_flat = self._frame2flat(self.train)

        self.test_nested = self._frame2nested(self.test)
        self.test_flat = self._frame2flat(self.test)
        for root, dirs, files in os.walk(maildir):
            stripped = root[len(maildir):]
            if cnt > limit + skip:
                break
            stripped = root[len(maildir):]
            log('DEBUG', "entering %s containing %d files and %d dirs" % (stripped, len(files), len(dirs)))

            for file in files:
                with open(root + "/" + file, "r", encoding='utf-8', errors='ignore') as f:
                    cnt += 1
                    if cnt < skip:
                        continue
                    # data += parser.parsestr(f.read()).get_payload() + '\n'
                    data += re.sub(r'[a-z]', 'x', parser.parsestr(f.read()).get_payload() + '\n', flags=re.IGNORECASE)