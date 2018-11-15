from __future__ import print_function
from __future__ import division

from sys import exit
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from collections import Counter

import numpy as np


def mynumbers(pred, y):
    recall = recall_score(y, pred)
    precision = precision_score(y, pred)
    f1 = f1_score(y, pred)
    acc = accuracy_score(y, pred)
    return (recall, precision, f1, acc)


class SentimentClassifier(object):
    def __init__(self, pos_sents, neg_sents):
        self.pos_sents = SentimentClassifier.readlines(pos_sents)
        self.neg_sents = SentimentClassifier.readlines(neg_sents)

    @staticmethod
    def readlines(myfile):
        with open(myfile, "r") as f:
            return [line.strip() for line in f.readlines()]


class ruleSentimentClassifier(SentimentClassifier):
    def __init__(self, pos_sents, neg_sents):
        super(ruleSentimentClassifier, self).__init__(pos_sents, neg_sents)
        self.vader = SentimentIntensityAnalyzer()
        print ("Finish reading sentences and intialize VADER ...")
        self.compute_score()

    def compute_score(self):
        retriver = lambda x: self.vader.polarity_scores(x)["compound"]
        _pos_preds = map(lambda x: int(x>=0.05), map(retriver, self.pos_sents))
        _neg_preds = map(lambda x: int(x<=-0.05), map(retriver, self.neg_sents))
        y = [1] * len(_pos_preds) + [0] * len(_neg_preds)
        result = mynumbers(_pos_preds+_neg_preds, y)
        print ("-" * 10 + " Summary (rule) " + "-" * 10)
        print ("Classification results:")
        print ("F1: {:.2f}, Recall: {:.2f}, Precision: {:.2f}, Accuracy: {:.2f}".format(
        result[2], result[0], result[1], result[-1]))
        print ("Data distribution: {}".format(dict(Counter(y))))
        print ("-" * 36)


if __name__ == "__main__":
    pos_sents = "./dataset/pos.txt"
    neg_sents = "./dataset/neg.txt"
    cls = ruleSentimentClassifier(pos_sents, neg_sents)
