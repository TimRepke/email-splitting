#!/usr/bin/python

from email import parser as ep
import os
from optparse import OptionParser
import arango
from joblib import Parallel, delayed
import multiprocessing
from collections import Counter
from itertools import repeat
from tqdm import tqdm

from logger import log, loglevel, is_lower


class Pipeline:
    def __init__(self):
        self.pipeline = []
        self._is_prepared = False

    @property
    def is_prepared(self):
        return self._is_prepared

    def add(self, step):
        self.pipeline.append(step)

    def prepare(self):
        for pipeline_part in self.pipeline:
            if not pipeline_part.is_prepared:
                log('DEBUG', 'Preparing pipeline part of type %s', type(pipeline_part))
                pipeline_part.prepare()
        self._is_prepared = True

    def transform(self, raw_mail, transformed=None):
        if not self.is_prepared:
            log('WARN', 'Forgot to prepare pipeline before using. Doing it for you!')
            self.prepare()

        if not transformed:
            transformed = raw_mail.get_payload()
        for pipeline_part in self.pipeline:
            log('TRACE', 'Calling pipeline part %s', type(pipeline_part))
            transformed = pipeline_part.transform(raw_mail, transformed)
        return raw_mail, transformed


class SourceFiles:
    def __init__(self, maildir, limit=None, skip=0):
        self.maildir = maildir
        self.mailparser = ep.Parser()
        self.limit = limit
        self.current_root = ''
        self.current_stripped = ''
        self.skip = skip

    def __iter__(self):
        self.run = 0
        self.os_walker = os.walk(self.maildir)
        self.current_dirs = []
        self.current_files = iter([])
        log('INFO', 'Created Source Iterator. It will skip %d files, then read %d files from %s',
            self.skip, self.limit, self.maildir)

        for i in range(self.skip):
            self.run += 1
            self._next_file(skipmode=True)
        return self

    def _next_dir(self):
        self.current_root, self.current_dirs, files = next(self.os_walker)
        self.current_stripped = self.current_root[len(self.maildir):]
        log('INFO', 'Entering directory with %d files and %d subdirectories: %s/',
            len(files), len(self.current_dirs), self.current_stripped)

        if len(files) > 0:
            self.current_files = iter(files)
        else:
            self._next_dir()

    def _next_file(self, skipmode=False):
        try:
            filename = next(self.current_files)
            log('DEBUG', 'Iterator is looking at this file (skip=%s): %s/%s', skipmode, self.current_root, filename)

            # save some effort when result is dumped anyway during skip-ahead
            if not skipmode:
                with open(self.current_root + "/" + filename, "r", errors='ignore') as f:
                    self.run += 1
                    file = f.read()

                    # must be something off here, skipping
                    if len(file) < 100:
                        log('WARNING', 'Skipping file because it is too small (%d chars): %s/%s',
                            len(file), self.current_root, filename)
                        return self._next_file()

                    return self.current_stripped, filename, self.mailparser.parsestr(file)
        except StopIteration:
            self._next_dir()
            log('DEBUG', 'Iterator had to switch directories')
            return self._next_file()

    def __next__(self):
        if self.limit is not None and (self.limit + self.skip) <= self.run:
            log('INFO', 'max number of mails (LIMIT=%d) is reached.', self.limit)
            raise StopIteration()

        return self._next_file()


class SourceArango:
    def __init__(self, user, pw, port=8529, db='enron', logging=True, skip=0, limit=None, collection='mails'):
        self.client = arango.ArangoClient(
            protocol='http',
            host='localhost',
            port=port,
            username=user,
            password=pw,
            enable_logging=logging
        )
        self.mailparser = ep.Parser()
        self.db = self.client.database(db)
        self.skip = skip
        self.limit = limit
        self.collection = collection
        self.count = None

    def __iter__(self):
        self.run = 0
        self.cursor = self.db.aql.execute('FOR m IN ' + self.collection + ' RETURN m', count=True)
        print('count', next(self.cursor))
        log('DEBUG', self.cursor.statistics())
        self.count = self.cursor.count() - self.skip
        log('DEBUG', 'ArangoSource has %d items, skipping %d, limiting to %d', self.cursor.count(), self.skip, self.limit or -1)
        for i in range(self.skip):
            self.run += 1
            next(self.cursor)

        return self

    def __len__(self):
        return self.count

    def __next__(self):
        if self.limit is not None and self.run >= (self.limit + self.skip):
            raise StopIteration

        self.run += 1
        doc = next(self.cursor)

        if self.collection == 'mails':
            head = '\n'.join(['%s: %s' % (k, v) for k, v in doc['header_raw'].items()])
            mail = self.mailparser.parsestr('%s\r\n%s' % (head, doc['body_raw']))

            return mail, doc
        else:
            return doc


class DataSinkArango:
    def __init__(self, user, pw, port=8529, db='enron', collection='mails', save=True, logging=True):
        self.client = arango.ArangoClient(
            protocol='http',
            host='localhost',
            port=port,
            username=user,
            password=pw,
            enable_logging=logging
        )
        try:
            self.client.create_database(db)
        except arango.ArangoError:
            pass
        try:
            self.client.database(db).create_collection(collection)
        except arango.ArangoError:
            pass
        try:
            self.client.database(db).create_collection('users')
        except arango.ArangoError:
            pass

        self.collection = self.client.database(db).collection(collection)

        self.save = save
        log('INFO', 'DataSink connects to port %d for collection "%s" in database "%s", active: %s',
            port, collection, db, save)

    def push(self, doc):
        if self.save:
            d = self.collection.insert(doc)

    def update(self, doc):
        if self.save:
            self.collection.update(doc)


oparser = OptionParser()
oparser.add_option("-d", "--maildir",
                   dest="maildir",
                   metavar="DIR",
                   type="str",
                   help="starting at the root of DIR, all subfolders are read recursively and "
                        "files are interpreted by the pipeline into a sink",
                   default='/home/tim/Uni/HPI/workspace/enron/data/original')
oparser.add_option("-n", "--limit",
                   dest="limit",
                   metavar="NUM",
                   help="limit number of mails to read to NUM",
                   type='int',
                   default=None)
oparser.add_option("-s", "--skip",
                   dest="skip",
                   metavar="NUM",
                   help="number (NUM) of mails to skip reading",
                   type='int',
                   default=0)
oparser.add_option("--keras-model",
                   dest="keras_model",
                   metavar="FILE",
                   type="str",
                   help="path to FILE where weights of neural net for splitting mails is/should be stored",
                   default='/home/tim/Uni/HPI/workspace/enron/pipeline/model.hdf5')
oparser.add_option("--retrain-keras",
                   dest="keras_retrain",
                   help="set this flag to train the neural net for splitting mails",
                   action="store_true")
oparser.add_option("--include-signatures",
                   dest="include_signature",
                   help="set this flag to include signature detection, empty fields are added either way!",
                   action="store_true")
oparser.add_option("--path-annotated",
                   dest="path_annotated",
                   metavar="DIR",
                   type="str",
                   help="path to the DIR containing annotated emails to train neural net for splitting mails",
                   default='/home/tim/Uni/HPI/workspace/enron/pipeline/annotated_mails/')
oparser.add_option("-u", "--db-user",
                   dest="arango_user",
                   metavar="USER",
                   type="str",
                   help="username to the arangodb",
                   default='root')
oparser.add_option("--db-pw",
                   dest="arango_pw",
                   metavar="PW",
                   type="str",
                   help="user password to the arangodb",
                   default='test')
oparser.add_option("--db-port",
                   dest="arango_port",
                   metavar="NUM",
                   type="int",
                   help="the port on localhost the arangodb is listening on",
                   default=8529)
oparser.add_option("--db-name",
                   dest="arango_db",
                   metavar="DB",
                   type="str",
                   help="name of the database (DB) to use",
                   default='enron')
oparser.add_option("--db-collection",
                   dest="arango_collection",
                   metavar="COLL",
                   type="str",
                   help="name of the collection (COLL) in the database (DB) to use",
                   default='mails')
oparser.add_option("--log-file",
                   dest="log_file",
                   metavar="FILE",
                   type="str",
                   help="log is written to FILE according to selected log LEVEL",
                   default=None)
oparser.add_option("--log-file-level",
                   dest="log_file_level",
                   metavar="LEVEL",
                   help="set log level to LEVEL",
                   type='choice',
                   choices=['MICROTRACE', 'TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'],
                   default='INFO')
oparser.add_option("-l", "--log-level",
                   dest="log_level",
                   metavar="LEVEL",
                   help="set log level to LEVEL",
                   type='choice',
                   choices=['MICROTRACE', 'TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'],
                   default='INFO')
oparser.add_option("-p", "--pipeline",
                   dest="pipeline",
                   metavar="NAME",
                   help="select which pipeline to use",
                   type='choice',
                   choices=['import', 'fix', 'NER', 'exp', 'NER2'])
oparser.add_option("--db-save",
                   dest="arango_no_save",
                   help="set this flag to disable the database sink!",
                   action="store_true",
                   default=False)


class NERWorker:
    def __init__(self, options, cores=4):
        self.data_source = SourceArango(options.arango_user, options.arango_pw, options.arango_port,
                                        options.arango_db, skip=options.skip, limit=options.limit, collection='sent')

        self.pipeline = Pipeline()
        self.pipeline.add(NamedEntityRecognition())
        self.pipeline.prepare()

        self.mail_sink = DataSinkArango(options.arango_user, options.arango_pw, options.arango_port,
                                        options.arango_db, options.arango_collection, save=options.arango_no_save)
        self.edge_sink = DataSinkArango(options.arango_user, options.arango_pw, options.arango_port,
                                        options.arango_db, options.arango_collection, save=options.arango_no_save)

        self.cores = cores

    @staticmethod
    def _work(mail, doc, data_sink, pipeline):
        log('TRACE', 'Got mail from source: %s', doc['_id'])
        mail, transformed = pipeline.transform(mail, doc['parts'])

        doc['parts'] = transformed
        data_sink.update(doc)
        return transformed

    def work(self):
        if self.cores == 1:
            l = []
            for (mail, doc), sink, line in zip(self.data_source,
                                               repeat(self.data_sink),
                                               repeat(self.pipeline)):
                l += NERWorker._work(mail, doc, sink, line)

        else:
            results = Parallel(n_jobs=self.cores)(delayed(NERWorker._work)(mail, doc, line)
                                                  for (mail, doc), line in zip(self.data_source, repeat(self.pipeline)))

            l = [ii for i in results for ii in i]

        pprint(Counter([t[0] for t in l]))
        pprint(Counter([t[1] for t in l if t[0] == 'ORG']).most_common(100))  # t[0] == 'PERSON' or
        print('Total unique Entities: ' + str(len(set([t[1] for t in l]))))


if __name__ == "__main__":
    (options, args) = oparser.parse_args()

    from logger import init as logging_init

    import io
    from pprint import pprint

    logging_init(options)

    if options.pipeline == 'import':
        from splitting_feature_rnn import Splitter
        from mixins import BodyCleanup, Tuples2Dicts
        from header_parsing_rules import ParseHeaderComponents, ParseAuthors, ParseDate

        data_source = SourceFiles(options.maildir, skip=options.skip, limit=options.limit)

        pipeline = Pipeline()
        pipeline.add(Splitter(options.path_annotated, window_size=8, include_signature=options.include_signature,
                              features=None, training_epochs=10, nb_slack_lines=4, retrain=options.keras_retrain,
                              model_path=options.keras_model))
        pipeline.add(Tuples2Dicts())
        pipeline.add(ParseHeaderComponents())
        pipeline.add(ParseAuthors())
        pipeline.add(ParseDate())
        pipeline.add(BodyCleanup())

        data_sink = DataSinkArango(options.arango_user, options.arango_pw, options.arango_port,
                                   options.arango_db, options.arango_collection, save=options.arango_no_save)

        read_cnt = 0

        for path, filename, mail in data_source:
            log('TRACE', 'Got mail from source: %s/%s', path, filename)
            mail, transformed = pipeline.transform(mail)

            if is_lower('TRACE'):
                tmp = io.StringIO()
                pprint(transformed, stream=tmp)
                log('MICROTRACE', tmp.getvalue())

            maildoc = {
                "message_id": mail['Message-ID'],
                "folder": '/'.join(path.split('/')[2:]),
                "file": path + '/' + filename,
                "owner": path.split('/')[1],
                "header_raw": dict(mail.items()),
                "body_raw": mail.get_payload(),
                "parts": transformed
            }
            data_sink.push(maildoc)

    elif options.pipeline == 'fix':
        from header_parsing_rules import ParseHeaderComponents, ParseAuthors, ParseDate

        data_source = SourceArango(options.arango_user, options.arango_pw, options.arango_port,
                                   options.arango_db, skip=options.skip, limit=options.limit)

        pipeline = Pipeline()
        pipeline.add(ParseAuthors())
        pipeline.prepare()

        data_sink = DataSinkArango(options.arango_user, options.arango_pw, options.arango_port,
                                   options.arango_db, options.arango_collection, save=options.arango_no_save)

        for mail, doc in data_source:
            log('TRACE', 'Got mail from source: %s', doc['_id'])
            mail, transformed = pipeline.transform(mail, doc['parts'])

            doc['parts'] = transformed
            data_sink.update(doc)

    elif options.pipeline == 'NER':
        from named_entity_recognition import NamedEntityRecognition

        data_source = SourceArango(options.arango_user, options.arango_pw, options.arango_port,
                                   options.arango_db, skip=options.skip, limit=options.limit, collection='sent')

        pipeline = Pipeline()
        pipeline.add(NamedEntityRecognition())
        pipeline.prepare()

        mail_sink = DataSinkArango(options.arango_user, options.arango_pw, options.arango_port,
                                   options.arango_db, options.arango_collection, save=options.arango_no_save)
        edge_sink = DataSinkArango(options.arango_user, options.arango_pw, options.arango_port,
                                   options.arango_db, options.arango_collection, save=options.arango_no_save)

        l = []
        for mail in data_source:
            maildoc = data_source.db.collection('mails').get(mail['mail_ids'][0].replace('mails/', ''))
            log('TRACE', 'Got mail from source: %s', maildoc['_id'])
            mail, transformed = pipeline.transform(maildoc['parts'], maildoc['parts'])

            l += transformed
            a = True
            for t in transformed:
                if 'enron' in t[1].lower() and a:
                    a = False
                    print('====================================================')
                    print('====================================================')
                    print(transformed)
                    for p in maildoc['parts']:
                        print(p['body'])
                        print('------------------------------------------------------')

                        # print({'name': t[1], 'type': t[0], 'mail': maildoc['_id']})

        pprint(Counter([t[0] for t in l]))
        # pprint(Counter([t[1] for t in l if t[0] == 'ORG']).most_common(100))  # t[0] == 'PERSON' or
        print('Total unique Entities: ' + str(len(set([t[1] for t in l]))))
    elif options.pipeline == 'NER2':
        from named_entity_recognition import NamedEntityRecognition

        data_source = SourceArango(options.arango_user, options.arango_pw, options.arango_port,
                                   options.arango_db, skip=options.skip, limit=options.limit, collection='sent')

        pipeline = Pipeline()
        pipeline.add(NamedEntityRecognition())
        pipeline.prepare()

        mail_sink = DataSinkArango(options.arango_user, options.arango_pw, options.arango_port,
                                   options.arango_db, options.arango_collection, save=options.arango_no_save)

        mids = []
        with open('../vis/routes/cache/raw_ents.json', 'w') as dump:
            dump.write('[')
            for mail in tqdm(data_source):
                if mail['mail_ids'][0] not in mids:
                    mids.append(mail['mail_ids'][0])
                    maildoc = data_source.db.collection('mails').get(mail['mail_ids'][0].replace('mails/', ''))
                    log('TRACE', 'Got mail from source: %s', maildoc['_id'])
                    mail, transformed = pipeline.transform(maildoc['parts'], maildoc['parts'])

                    for t in transformed:
                        dump.write(str({
                            "mid": maildoc['_id'],
                            "entity": t[1],
                            "type": t[0],
                            "part": t[2]
                        }))
                        dump.write(',\n')
                else:
                    log('DEBUG', 'skip '+mail['mail_ids'][0])
            dump.write(']')

#cat raw_ents.json | tr "'" '"' > tmp && mv tmp raw_ents.json
#sed -i -r -e "s/: \"([^\",]+)\"([^\",]+)\",/: \"\1'\2\",/g" raw_ents.json


    elif options.pipeline == 'exp':
        def iv(t):
            return t.is_alpha


        from spacy_singleton import get_nlp_model

        nlp = get_nlp_model()
        data_source = SourceArango(options.arango_user, options.arango_pw, options.arango_port,
                                   options.arango_db, skip=options.skip, limit=options.limit, collection='sent')
        l = []
        r = []
        k = 0
        for mail in data_source:
            maildoc = data_source.db.collection('mails').get(mail['mail_ids'][0].replace('mails/', ''))
            for p in maildoc['parts']:
                s = nlp(p['body'])
                for e in s.ents:
                    if e.root.ent_type_ == 'PERSON':
                        i = e.root.i
                        if i > k:
                            if iv(s[i - 1 - k]):
                                l.append(s[i - 1 - k])
                        if i < len(s) - 1 - k:
                            if iv(s[i + 1 + k]):
                                r.append(s[i + 1 + k])
        print(['>"' + str(ll) + '"<' for ll in l])
        print(['>"' + str(rr) + '"<' for rr in r])
        print(Counter([str(ll) for ll in l]).most_common(40))
        print(Counter([str(rr) for rr in r]).most_common(40))


        # print('========================================= Got mail from source: %s', maildoc['_id'])
        # for p in maildoc['parts']:
        #     print(p['body'])
        #     print('-------......---------......--------......------.....')
