from email import parser as ep
from datetime import datetime, timezone
import os
import re


class Email:
    def __init__(self, path, fn, mail):
        self.path = path
        self.fn = fn
        self.mail = mail

    @property
    def sent(self):
        return datetime.strptime(re.sub(r' *\([A-Z]+\)', '', self.mail['Date']),
                                 '%a, %d %b %Y %H:%M:%S %z').astimezone(timezone.utc)

    @property
    def file(self):
        return self.path + '/' + self.fn

    @property
    def folder(self):
        return '/'.join(self.path.split('/')[1:])

    @property
    def id(self):
        return self.mail['Message-ID']

    @property
    def mailbox(self):
        return self.mail['X-Origin']

    @property
    def body(self):
        return self.mail.get_payload()

    @property
    def subject(self):
        return self.mail['Subject']

    @property
    def sender(self):
        return self.mail['From']

    @property
    def xsender(self):
        return self.mail['X-From']

    @property
    def to(self):
        return self.mail['To']

    @property
    def xto(self):
        return self.mail['X-To']

    @property
    def cc(self):
        return self.mail['Cc']

    @property
    def xcc(self):
        return self.mail['X-cc']

    @property
    def bcc(self):
        return self.mail['Bcc']

    @property
    def xbcc(self):
        return self.mail['X-bcc']


class EmailFiles:
    def __init__(self, maildir, limit=None, skip=0):
        self.maildir = maildir
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

                    return self.current_stripped, filename, file
        except StopIteration:
            self._next_dir()
            return self._next_file()

    def __next__(self):
        if self.limit is not None and (self.limit + self.skip) <= self.run:
            raise StopIteration()

        return self._next_file()


class Emails(EmailFiles):
    def __init__(self, maildir, limit=None, skip=0):
        super().__init__(maildir, limit, skip)
        self.mail_parser = ep.Parser()

    def __next__(self):
        path, fn, file = super().__next__()
        return Email(path, fn, self.mail_parser.parsestr(file))
