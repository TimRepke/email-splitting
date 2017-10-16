import re

results = []
models = ['cnn', 'rnn']
with open('../DeepRNN/eval_line_embedding.log', 'r') as f:
    line_len = 0
    num_labels = 0
    embedding_size = 0
    tmp = {}
    cnt = 0
    for line in f:
        if re.match('^line', line):
            values = list(re.findall('\d+', line))
            tmp = {
                'line_len': int(values[0]),
                'n_lab': int(values[1]),
                'emb': int(values[2])
            }

        if re.match('^Accuracy', line):
            tmp['acc'] = float(line.strip().split(' ')[2])

        if re.match('^ *avg', line):
            values = list(re.findall('\d+\.\d+', line))
            tmp['prec'] = float(values[0])
            tmp['rec'] = float(values[1])
            tmp['f1'] = float(values[2])

            tmp['model'] = models[cnt % 2]
            cnt += 1
            results.append(dict(tmp))

    # print(len(results))
    # for r in results:
    #    print(r)

    # csv format:
    # line_len (prec|rec|f1|acc)_<num_labels>_<embedding>_<model>
    trans = {}
    for metric in ['line_len', 'prec', 'rec', 'f1', 'acc']:
        for r in results:
            key = metric + '-' + str(r['n_lab']) + '-' + str(r['emb']) + '-' + r['model']
            key = key.replace('-', '').replace('_', '')
            if key not in trans:
                trans[key] = []
            trans[key].append(r[metric])

    cols = list(trans.keys())
    print(list(enumerate(cols)))
    print('\t'.join(cols))
    for row in zip(*list(trans.values())):
        print(','.join([str(r) for r in row]))
