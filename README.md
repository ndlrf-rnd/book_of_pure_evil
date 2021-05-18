# book_of_pure_evil
Main tasks of project:
1. extremism text classification
2. text generation with GAN/RL or both
3. book ideas visualization

## algo results

1-s, 3-s, 5-s means that we split training and test data by 1 sentence, 3 sentences, 5 sentences accordingly 

### word2vec + Random Forest classification; metric: ROC-AUC
| Model/Split               |  1-s  |  3-s  |  5-s  |
|:--------------------------|:------|:------|:------|
| our_corpus (2 epochs)     | 0.55  | 0.58  | 0.57  | 
| rusvectores               | 0.71  | 0.69  | 0.67  | 

### fasttext (train_supervised method); metric: PRECISION and RECALL (they are the same)
| Model/Split               |  1-s  |  3-s  |
|:--------------------------|:------|:------|
| our_corpus (2 epochs)     | 0.88  | 0.95  |
| rusvectores               | 0.87  | 0.94  |
| baseline (w/o pretrain)   | 0.88  | 0.94  |
