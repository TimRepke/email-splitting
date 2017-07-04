## build index
see script in scripts folder

 takes about 30min
```
$ tree -d enron/data/original | wc -l
52903
```
the index may contain duplicates. to remove them, run
```
$ sqlite3 mails.db 'SELECT a.id, b.id, a.sender, b.sender, (julianday(a.date)-julianday(b.date))*24 as diff, a.`to`, b.`to` FROM enron_mail a, enron_mail b WHERE a.id<b.id and a.split is not null and b.split is not null and abs(diff)<0.007' > results.txt
$ cat results.txt | cut -d '|' -f2 | grep "^[0-9]\+$" | sort -n | uniq -u | awk '{ print "UPDATE enron_mail SET exclude=1 WHERE id="$1";" }' > updates.sql
$ sqlite3 mails.db updates.sql
```

## Train/Test/Eval
Set for training, development testing and final evaluation are first split by mailbox as follows:

- **TRAIN:** ("pimenov-v","scholtes-d","arora-h","mims-p","mckay-b","mccarty-d","ring-a","wolfe-j","geaccone-t","parks-j","whalley-l","williams-w3","buy-r","dean-c","hernandez-j","giron-d","sager-e","haedicke-m","kitchen-l","lokay-m","rogers-b","scott-s","nemec-g","symes-k","beck-s","shackleton-s","jones-t","dasovich-j") 
- **TEST:** ("rapp-b","gang-l","staab-t","townsend-j","may-l","lewis-a","horton-s","skilling-j","sanders-r","germany-c","mann-k","kaminski-v")
- **EVAL:**("gilbertsmith-d","weldon-v","semperger-c","pereira-s","forney-j","heard-m","presto-k","grigsby-m","davis-d","mcconnell-m","bass-e","farmer-d","taylor-m","kean-s")

```
select split as ns, count(1) as cnt from enron_mail group by split order by cnt
```

| Split | Count    |
|-------|----------|
| "2"   | "76295"  |
| "1"   | "83946"  |
| "0"   | "165312" |
| NULL  | 191448   |
