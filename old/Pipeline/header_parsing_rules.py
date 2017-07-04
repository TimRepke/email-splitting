import re
from logger import log
from dateutil.parser import parse as dateparser


class ParseHeaderComponents:
    @property
    def is_prepared(self):
        return True

    def prepare(self, *args, **kwargs):
        pass

    def _clean_head(self, s):
        return s.replace('\n', '')

    def _kw2key(self, kw):
        kw = kw.lower().strip()
        if kw == 'from':
            return 'from'
        if kw == 'to':
            return 'to'
        if kw == 'cc':
            return 'cc'
        if kw == 'bcc':
            return 'bcc'
        if kw == 'subj' or kw == 'subject':
            return 'subject'
        if kw == 'date' or kw == 'sent':
            return 'date'

    def _clean_subject(self, s):
        return re.sub(r"fw:?|re:?", '', s, flags=re.IGNORECASE).strip()

    def _transform_head(self, raw):
        # put default in all fields
        raw['head'] = {
            'from': '',
            'to': '',
            'cc': '',
            'bcc': '',
            'subject': '',
            'date': ''
        }
        # clean the extracted head
        head = self._clean_head(raw['head_raw'])

        keywords = re.finditer(r"(from|(?<!mail)to|cc|bcc|subj|subject|date|sent):", head,
                               re.IGNORECASE | re.DOTALL | re.VERBOSE)
        try:
            grp = next(keywords)
            kw = grp.group(1)
            kw_end = grp.end()

            for grp in keywords:
                txt = head[kw_end:grp.start()]
                raw['head'][self._kw2key(kw)] = txt.strip()
                kw = grp.group(1)
                kw_end = grp.end()
            # add dangling rest
            txt = head[kw_end:]
            raw['head'][self._kw2key(kw)] = txt.strip()

            if ' on ' in raw['head']['from'].lower() and not raw['head']['date']:
                tmp = raw['head']['from'].lower().split(' on ')
                raw['head']['from'] = tmp[0].strip()
                raw['head']['date'] = tmp[1].strip()

            raw['subject'] = self._clean_subject(raw['head']['subject'])
        except StopIteration:
            log('WARNING', 'Failed to split head into its parts!')

        return raw

    def transform(self, mail, processed):
        return [self._transform_head(p) for p in processed]


class ParseAuthors:
    @property
    def is_prepared(self):
        return True

    def prepare(self, *args, **kwargs):
        pass

    def _prepare_string(self, s):
        s = re.sub(r'\n', '', (s or ''))
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'((?:\w+ )+\w+),', '\g<1>;', (s or ''))
        s = re.sub(r'<(/\w+=[^/>]+)+>,', ';', (s or ''))
        s = re.sub(r'<(/\w+=[^/>]+)+>', '', (s or ''))
        s = re.sub(r'([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}) ?\[([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\]',
                   '\g<1> \g<2>', (s or ''), flags=re.IGNORECASE)
        s = re.sub(r"'([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})',", '\g<1>;', (s or ''), flags=re.IGNORECASE)
        s = re.sub(r"'([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})'", '\g<1>', (s or ''), flags=re.IGNORECASE)
        s = re.sub(r'.+on behalf of ((?:[A-Z0-9._%+-]| )+@[A-Z0-9.-]+(:?\.[A-Z]{2,})?)',
                   '\g<1>', (s or ''), flags=re.IGNORECASE)
        s = re.sub(r"'?([^',]+, ?[^',]+)'? <([^>]+)>,", '\g<1> \g<2>;', (s or ''), flags=re.IGNORECASE)
        s = re.sub(r"'?([^',]+, ?[^',]+)'? <([^>]+)>", '\g<1> \g<2>', (s or ''), flags=re.IGNORECASE)
        s = re.sub(r"'?([^',]+)'? ?<([^>]+)>", '\g<1> \g<2>', (s or ''), flags=re.IGNORECASE)
        s = re.sub(r"'?((?:[^', ]+){2,3})'? \[([^>]+)\]", '\g<1> \g<2>', (s or ''))
        s = re.sub(r'\\|"', '', (s or ''))
        s = re.sub(r'mailto:', '', (s or ''))
        s = s.strip().lower()
        s = re.sub(r'^\W*(.+?)\W*$', '\g<1>', (s or ''), flags=re.IGNORECASE)
        s = re.sub(r'^([^/]+)/.*$', '\g<1>', (s or ''), flags=re.IGNORECASE)
        s = re.sub(r'^"?([a-z\', ]+)', '\g<1>', (s or ''), flags=re.IGNORECASE)
        s = re.sub(r'\s+\d+/\d+/\d+\s*\d+:\d+\s*(am|pm)?', '', (s or ''), re.IGNORECASE | re.MULTILINE)

        return s.strip().lower()

    def _process_author(self, s):
        name = re.sub(r"[a-z0-9_\-.]+@[a-z0-9_\-.]+\.[a-z]+", '', (s or ''), flags=re.IGNORECASE).strip()
        name = re.sub(r"@.+$", '', name)
        name = re.sub(r"\(r\)|\.|\d|\(|\)|\[|\]|=|<|>", '', name)
        name = re.sub(r"e-mail", '', name)
        name = re.sub(r"\s+", ' ', name).strip()

        if ',' in name:  # fix format: surname, firstname
            name = ' '.join(reversed([sp.strip() for sp in name.split(',')]))

        mail = re.sub(r".*?([a-z0-9_\-.]+@[a-z0-9_\-.]+\.[a-z]+).*", '\g<1>', (s or ''),
                      flags=re.IGNORECASE).strip() if '@' in s else ''

        log('MICROTRACE', 'Processing s="%s" and extracted name="%s" and email="%s"', s, name, mail)

        return {
            'name': name,
            'email': mail
        }

    def _split_authors(self, s, kind):
        authors = self._prepare_string(s).split(';')
        ret = []
        for i, author in enumerate(authors):
            tmp = self._process_author(author)
            tmp['pos'] = i
            tmp['kind'] = kind
            if tmp['name']:
                ret.append(tmp)
        return ret

    def _transform_recipients(self, head):
        ret = []
        for kind in ['to', 'cc', 'bcc']:
            ret += self._split_authors(head[kind], kind)
        return ret

    def transform(self, mail, processed):
        for i, part in enumerate(processed):
            processed[i]['recipients'] = self._transform_recipients(part['head'])
            processed[i]['sender'] = self._process_author(self._prepare_string(part['head']['from']))

        return processed


class ParseDate:
    @property
    def is_prepared(self):
        return True

    def prepare(self, *args, **kwargs):
        pass

    def _parse(self, s):
        try:
            date = dateparser(s, fuzzy=True)
            return date.strftime('%Y/%m/%d %H:%M')
        except ValueError as e:
            log('WARN', 'Failed to parse date="%s" which triggered an error: %s', s, e)

    def transform(self, mail, processed):
        # TODO fallback to date from MIME (when not parseable or empty)
        # TODO add constraints on time to 1998-2003
        for i, part in enumerate(processed):
            processed[i]['date'] = self._parse(part['head']['date'] or '')
        return processed

# def parseUser(s):
#     # TODO: remove placeholders (i.e. no.address@)
#     address = '' if not s else s
#     address = address.replace('\n', '').replace('\t', '').strip()
#
#     # fetch address that has the format "e-mail <'name'bla.js@enron.com>"
#     m = re.search(r"e-mail <'?(.*?)['\.](.+?)@(.+?)>", address)
#     if m:
#         return {
#             "raw": s,
#             "name": m.group(1),
#             "mail": m.group(2) + '@' + m.group(3),
#             "domain": m.group(3)
#         }
#
#     address = repl(["'", '"', '<', '>'], address)
#
#     if len(address) > 0:
#         return {
#             "raw": s,
#             "name": '',
#             "mail": address,
#             "domain": '' if '@' not in address else address.split('@')[1]
#         }
#
#     return None
#
#
# def parseUsers(to, cc, bcc):
#     ret = []
#     for k, v in {'to': to, 'cc': cc, 'bcc': bcc}.items():
#         for i, a in enumerate(('' if not v else v).split(',')):
#             t = parseUser(a)
#             if t:
#                 t['kind'] = k
#                 t['pos'] = i
#                 ret.append(t)
#     return ret
#
