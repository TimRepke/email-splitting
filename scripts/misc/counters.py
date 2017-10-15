from scripts.utils import AnnotatedEmailsIterator
import numpy as np
from collections import Counter
from itertools import tee


def stat(splitit):
    splitit = list(splitit)
    headers = [len([1 for deno in mail.denotations if deno['type'] == 'Header']) for mail in splitit]
    print('num mails', len(headers))
    print('num mails - no thread', len([h for h in headers if h == 0]))
    print('num mails - with thread', len([h for h in headers if h > 0]))
    print('num actual messages', (np.array(headers)+1).sum())
    print('avg threadlen', (np.array(headers)+1).mean())
    print('avg threadlen - nonzero', (np.array([h for h in headers if h > 0])+1).mean())
    c = Counter(headers)
    total = sum(c.values(), 0.0)
    print('num headers per mail', ', '.join(['%d: %.2f' % (k, c[k] / total) for k in sorted(c)]))
    print('num sigs', np.array(([l for l in [len([1 for deno in mail.denotations if deno['type'] == 'Body/Signature'])
                                       for mail in splitit] if l > 0])).sum())
    print('avg sig len', np.array([len(deno['text'].split('\n')) for mail in splitit for deno in mail.denotations if
                                   deno['type'] == 'Body/Signature']).mean())

    print('avg email len (lines)', np.array([len(m.body.split('\n')) for m in splitit]).mean())
    print('avg message len (lines)', np.array([len(deno['text'].split('\n')) for mail in splitit
                                       for deno in mail.denotations if deno['type'] == 'Body']).mean())
    print('avg line len:', np.array([len(line) for mail in splitit for line in mail.lines]).mean())
    print('avg line len (w/o empty lines):', np.array([l for l in [len(line.strip()) for mail in splitit for line in mail.lines] if l > 1]).mean())

    print('mails with advanced annotation', len([]))


if __name__ == '__main__':

    for directory in ["../../data/enron/annotated", "../../data/asf/annotated"]:  # "../../data/enron/annotated_full",
        print('\n##### > ' + directory.upper() + ' < #####')
        iterator = AnnotatedEmailsIterator(directory)

        print('=========== Train:')
        stat(iterator.train)

        print('=========== Test:')
        stat(iterator.test)

        print('=========== Eval:')
        stat(iterator.eval)
