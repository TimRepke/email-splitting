import time
from IPython.display import clear_output


def printpred(tmpX, preds, annot):
    print("---------------------")
    for li, (n, line) in enumerate(tmpX.iterrows()):
        try:
            print(str(li).rjust(3) + ". (" + (", ".join(["%.3f" % p for p in preds[li]])) + ") " + \
                  {'H': 'H--', 'B': '-B-', 'S': '--S'}[annot[li]] + ">" + line['text'])
        except KeyError:
            print("     (X.XXX, X.XXX) --->" + line['text'])
    print("================================")


stop = False
for i, fname in enumerate(listdir(mailpathMore)):
    print(str(i) + ') checking', fname)
    if isfile(join(mailpathMore, fname)) and not isfile(join(mailpathAnno, fname)):
        with open(join(mailpathMore, fname)) as file:
            try:
                print("reading file...")
                origFile = file.read()
                mail = parser.parsestr(origFile)

                print("creating frame and vectors...")
                tmpX = toFrame([cleanbody(mail.get_payload())], [mail], [['b'] * len(mail.get_payload().splitlines())])
                tmpXm, tmpY, tmpI = toNested(tmpX, windowsize=linesAtOnce, caty=True, amap=annotationMap)

                print("predicting...")
                tmpy_pred = model.predict_proba(tmpXm, verbose=0)

                print("merging predictions...")
                preds = {}
                for wi, window in enumerate(tmpI):
                    for mj, maili in enumerate(window):
                        if maili not in preds:
                            preds[maili] = []
                        preds[maili].append(tmpy_pred[wi][mj])

                print("flattening prediction...")
                predflat = []
                predflat_p = []
                for li, (n, line) in enumerate(tmpX.iterrows()):
                    tmp = np.nanmean(preds[n], axis=0)
                    tmpa = np.nanargmax(tmp)
                    predflat.append(tmpa)
                    predflat_p.append(tmp)

                annot = [{0: 'H', 1: 'B', 2: 'S'}[pfi] for pfi in predflat]
                f = t = 0
                lastmsg = ""
                while True:
                    printpred(tmpX, predflat_p, annot)
                    print(lastmsg)
                    a = input()
                    if not a or a == "":
                        print("empty input received!")
                        time.sleep(1)
                        continue
                    elif a == "skip":
                        print("skipping " + fname)
                        time.sleep(2)
                        clear_output()
                        break
                    elif a == "stop":
                        print("stopping!")
                        stop = True
                        time.sleep(1)
                        break
                    elif a == "ok":
                        mail.set_payload(
                            "\n".join([ann + ">" + lin for ann, lin in zip(annot, mail.get_payload().splitlines())]))
                        print(mail.as_string())
                        print('===')

                        with open(join(mailpathAnno, fname), "w") as ofil:
                            ofil.write(mail.as_string())
                            ofil.close()

                        time.sleep(2)
                        clear_output()
                        break
                    elif re.match(r"(?P<anno>H|B|S)(?P<from>\d+)?\-?(?P<to>\d+)?", a, flags=re.I):
                        m = re.match(r"(?P<anno>H|B|S)(?P<from>\d+)?\-?(?P<to>\d+)?", a, flags=re.I)
                        ann = (m.group('anno') or "B").upper()
                        if m.group('from') and m.group('to'):
                            f = int(m.group('from'))
                            t = int(m.group('to'))
                        elif m.group('from') and not m.group('to'):
                            f = t + 1 if t > 0 else 0
                            t = int(m.group('from'))
                        else:
                            print("didn't get that...")
                            time.sleep(1)
                            clear_output()
                            continue
                        lastmsg = 'Updated lines ' + str(f) + ' to ' + str(t) + ' as ' + ann + '>'
                        for fti in range(f, t + 1):
                            annot[fti] = ann
                    else:
                        print("nope.")

                    clear_output()
            except Exception as e:
                print("ERROR OCCURED! SKIPPING!")
                print(e)
                # raise e

            finally:
                file.close()

            if stop:
                print('break')
                break