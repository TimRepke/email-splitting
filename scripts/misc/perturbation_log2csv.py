import re

models = ['featrnn', 'jangada', 'zebra']
datasets = ['asf', 'enron']
zoness = [2, 5]
results = []

for model in models:
    for dataset in datasets:
        for zones in zoness:
            with open('../../data/results/' + model + '_results_' + dataset + '_' + str(zones) + 'zones.log', 'r') as f:
                line_len = 0
                num_labels = 0
                embedding_size = 0
                tmp = {
                    'model': model,
                    'dataset': dataset,
                    'zones': zones
                }
                cnt = 0
                for line in f:
                    if re.match('^PERTUR', line):
                        tmp['pert'] = float(line.split(' ')[1])

                    prefix = '' if (cnt % 2) == 0 else 'w'
                    if re.match('^Accuracy', line):
                        tmp[prefix + 'acc'] = float(list(re.findall('\d+\.\d+', line))[0])

                    if re.match('^ *avg', line):
                        values = list(re.findall('\d+\.\d+', line))
                        tmp[prefix + 'prec'] = float(values[0])
                        tmp[prefix + 'rec'] = float(values[1])
                        tmp[prefix + 'f1'] = float(values[2])

                        if (cnt % 2) == 1:
                            results.append(dict(tmp))
                        cnt += 1

print(len(results))
for r in results:
    print(r)

# csv format:
# perturbation (w)(prec|rec|f1|acc)_<model>_<dataset>_<zones>
trans = {}
for metric in ['prec', 'rec', 'f1', 'acc', 'wprec', 'wrec', 'wf1', 'wacc']:
    for r in results:
        key = metric + '-' + r['model'] + '-' + r['dataset'] + '-' + str(r['zones'])
        # key = key.replace('-', '').replace('_', '')
        if key not in trans:
            trans[key] = []
        trans[key].append(r[metric])

cols = ['perturbation'] + list(trans.keys())
print(list(enumerate(cols)))
print('\t'.join(cols))
vals = [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
vals += list(trans.values())
for row in zip(*vals):
    print(','.join([str(r) for r in row]))
