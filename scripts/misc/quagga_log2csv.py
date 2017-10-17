import re

datasets = ['asf', 'enron']
zoness = [2]#, 5]
results = []

for dataset in datasets:
    for zones in zoness:
        with open('../../data/results/quagga' + '_results_' + dataset + '_' + str(zones) + 'zones.log', 'r') as f:
            line_len = 0
            num_labels = 0
            embedding_size = 0
            tmp = {
                'dataset': dataset,
                'zones': zones
            }
            cnt = 0
            for line in f:
                if re.match('-* TEST -*', line):
                    subset = 'test'
                if re.match('-* EVAL -*', line):
                    subset = 'eval'

                if re.match('^PERTUR', line):
                    tmp['pert'] = float(line.split(' ')[1])

                prefix = '' if (cnt % 2) == 0 else 'w'
                if re.match('^Accuracy', line):
                    tmp[subset + '_' + prefix + 'acc'] = float(list(re.findall('\d+\.\d+', line))[0])

                if re.match('^ *avg', line):
                    values = list(re.findall('\d+\.\d+', line))
                    tmp[subset + '_' + prefix + 'prec'] = float(values[0])
                    tmp[subset + '_' + prefix + 'rec'] = float(values[1])
                    tmp[subset + '_' + prefix + 'f1'] = float(values[2])

                    if (cnt % 2) == 1 and subset=='eval':
                        results.append(dict(tmp))
                    cnt += 1

print(len(results))
for r in results:
    print(r)

# csv format:
# perturbation (test|eval)_(w)(prec|rec|f1|acc)_<dataset>_<zones>
trans = {}
for metric in ['prec', 'rec', 'f1', 'acc']:
    for subset in ['test', 'eval']:
        for prefix in ['', 'w']:
            for r in results:
                key = subset + '-' + prefix + '-' + metric + '-' + r['dataset'] + '-' + str(r['zones'])
                # key = key.replace('-', '').replace('_', '')
                if key not in trans:
                    trans[key] = []
                trans[key].append(r[subset + '_' + prefix + metric])

cols = ['perturbation'] + list(trans.keys())
print(list(enumerate(cols)))
print('\t'.join(cols))
vals = [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
vals += list(trans.values())
for row in zip(*vals):
    print(','.join([str(r) for r in row]))
