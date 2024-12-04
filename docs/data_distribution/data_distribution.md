Data distribution for Gait dataset
===

## Motation

When I analyzed the error predicted for the trained model, I found that the model had bad performance some fold, such as fold 0, 1, 2. But the model had good performance in fold 3, 4.

| Fold | validation loss | Accuracy | Precision | Recall | F1-score | AUROC  |
| ---- | --------------- | -------- | --------- | ------ | -------- | ------ |
| 0    | 1.82            | 0.7008   | 0.8551    | 0.5814 | 0.6922   | 0.7555 |
| 1    | 1.06            | 0.6735   | 0.6723    | 0.6219 | 0.6461   | 0.7050 |
| 2    | 1.17            | 0.6958   | 0.6133    | 0.4677 | 0.5307   | 0.7170 |
| 3    | 0.23            | 0.9077   | 0.9078    | 0.8596 | 0.8830   | 0.9594 |
| 4    | 0.81            | 0.8106   | 0.8155    | 0.7053 | 0.7564   | 0.8431 |
| Mean | Nan             | 0.7553   | 0.7850    | 0.6420 | 0.7063   | 0.7864 |
<center> The predict metrics of different fold. </center>

I do not know why this happened. 
So I want to find some way to analyze the data distribution of the dataset, then maybe helpfully to find the reason.

Here we decide to from different two ways to analyze the data distribution.

1. analyzed the difference between the different fold. Such as, splitted video number, patient number, etc.
2. Use some feature extract method to analyze the data distribution. Such as, PCA, TSNE, etc. 