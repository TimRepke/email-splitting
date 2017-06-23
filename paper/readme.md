# Notes

## Key Points/Goals
- Split Email conversations
- Extract and parse header information (meta data)
- identify parts of mails (greetings, signatures, disclaimers, ...)
- 

## data
- Ben Shneiderman's Email Archive (http://www.cs.umd.edu/hcil/ben-email-archive/)
- Enron Corpus
- Jeb Bush 
- 

## Train/Dev/Test split
- choose 5(?) inboxes for dev/test each
- sample mails from that
- remove all mails, that contains senders/recipients that are in train
- for train, randomly sample mails from all other inboxes
How many samples are enough?

## Evaluation
- cross corpus training (i.e. train on enron, test on shneiderman)
- introduce perturbation (i.e. randomly inserting chars, line breaks)
  - how is the accuracy influenced? 
  - already do that in training? maybe makes system more robust
- baselines
  - related work
  - rule based system (header lines start with "From:",...; signatures heuristically, i.e. contains sender name + all following)
  - already existing RNN with handcrafted features

## Conferences
- ECIR 2018, http://www.ecir2018.org/important-dates/ (proposals in Sept; full papers Oct; short papers Nov; notifications Jan; Conference 25-29 Mar)
- WSDM 2018, http://www.wsdm-conference.org/2018/call-for-papers.html (abstract 4. Aug; submission 11. Aug; notification 23. Oct; Conference 5-9 Feb)
- 


# LLNCS Instructions
Dear LLNCS user,

The files in this directory belong to the LaTeX2e package for
Lecture Notes in Computer Science (LNCS) of Springer-Verlag.

It consists of the following files:

  readme.txt         this file

  history.txt        the version history of the package

  llncs.cls          the LaTeX2e document class

  llncs.dem          the sample input file

  llncs.doc          the documentation of the class (LaTeX source)
  llncsdoc.pdf       the documentation of the class (PDF version)
  llncsdoc.sty       the modification of the class for the documentation
  llncs.ind          an external (faked) author index file
  subjidx.ind        subject index demo from the Springer book package
  llncs.dvi          the resultig DVI file (remember to use binary transfer!)

  sprmindx.sty       supplementary style file for MakeIndex
                     (usage: makeindex -s sprmindx.sty <yourfile.idx>)

  splncs03.bst       current LNCS BibTeX style with alphabetic sorting

  aliascnt.sty       part of the Oberdiek bundle; allows more control over
                     the counters associated to any numbered item
  remreset.sty       by David Carlisle
