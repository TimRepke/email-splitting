from email import parser as ep
from datetime import datetime, timezone
import os
import re


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

        for i in range(self.skip):
            self.run += 1
            self._next_file(skipmode=True)
        return self

    def _next_dir(self):
        self.current_root, self.current_dirs, files = next(self.os_walker)
        self.current_stripped = self.current_root[len(self.maildir):]
        if len(files) > 0:
            self.current_files = iter(files)
        else:
            self._next_dir()

    def _next_file(self, skipmode=False):
        try:
            filename = next(self.current_files)

            # save some effort when result is dumped anyway during skip-ahead
            if not skipmode:
                with open(self.current_root + "/" + filename, "r", errors='ignore') as f:
                    self.run += 1
                    file = f.read()

                    # must be something off here, skipping
                    if len(file) < 100:
                        return self._next_file()

                    return self.current_stripped, filename, self.mailparser.parsestr(file)
        except StopIteration:
            self._next_dir()
            return self._next_file()

    def __next__(self):
        if self.limit is not None and (self.limit + self.skip) <= self.run:
            raise StopIteration()

        return self._next_file()


from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class EnronMail(Base):
    __tablename__ = 'enron_mail'
    id = Column(Integer, primary_key=True)
    mid = Column(String)
    mailbox = Column(String)
    folder = Column(String)
    path = Column(String)
    sender = Column(String)
    xsender = Column(String)
    to = Column(String)
    xto = Column(String)
    cc = Column(String)
    xcc = Column(String)
    bcc = Column(String)
    xbcc = Column(String)
    subject = Column(String)
    date = Column(DateTime)

    def __repr__(self):
        return "<User(id=%s, from='%s[%s]', date='%s', path='%s')>" % (
            self.id, self.sender, self.xsender, self.date, self.path)


if __name__ == "__main__":
    # takes about 30min
    # > tree -d enron/data/original | wc -l
    # > 520903
    engine = create_engine('sqlite:///mails.db', echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    for i, (path, fn, mail) in enumerate(SourceFiles('../../enron/data/original/')):
        date = datetime.strptime(re.sub(r' *\([A-Z]+\)', '', mail['Date']), '%a, %d %b %Y %H:%M:%S %z').astimezone(timezone.utc)
        # print(path+fn, mail['Date'], date)
        session.add(EnronMail(mid=mail['Message-ID'], mailbox=mail['X-Origin'], folder='/'.join(path.split('/')[1:]),
                              path=path+'/'+fn,
                              sender=mail['From'], xsender=mail['X-From'],
                              to=mail['To'], xto=mail['X-To'],
                              bcc=mail['Cc'], xcc=mail['X-cc'],
                              cc=mail['Bcc'], xbcc=mail['X-bcc'],
                              subject=mail['Subject'],
                              date=datetime.strptime(re.sub(r' *\([A-Z]+\)',
                                                            '',
                                                            mail['Date']),
                                                     '%a, %d %b %Y %H:%M:%S %z').astimezone(timezone.utc)))
        if i % 1000 == 0:
            print('commit', i)
            session.commit()
