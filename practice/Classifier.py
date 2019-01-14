from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from collections import Counter
from svms import svmclsbinary
from overrides import overrides

from helpers import word2vec

import numpy as np


class SentimentClassifier(object):
    def __init__(self, pos_sents, neg_sents):
        self.pos_sents = SentimentClassifier.readlines(pos_sents)
        self.neg_sents = SentimentClassifier.readlines(neg_sents)

    @staticmethod
    def readlines(myfile):
        with open(myfile, "r") as f:
            return [line.strip() for line in f.readlines()]

    @staticmethod
    def mynumbers(pred, y):
        recall = recall_score(y, pred)
        precision = precision_score(y, pred)
        f1 = f1_score(y, pred)
        acc = accuracy_score(y, pred)
        return (recall, precision, f1, acc)


class ruleSentimentClassifier(SentimentClassifier):
    def __init__(self, pos_sents, neg_sents):
        super(ruleSentimentClassifier, self).__init__(pos_sents, neg_sents)
        self.vader = SentimentIntensityAnalyzer()
        print ("[INFO]: finish reading sentences and intialize VADER ...")
        self.compute_score()

    def compute_score(self):
        retriver = lambda x: self.vader.polarity_scores(x)["compound"]
        _pos_preds = map(lambda x: int(x>=0.05), map(retriver, self.pos_sents))
        _neg_preds = map(lambda x: int(x<=-0.05), map(retriver, self.neg_sents))
        y = [1] * len(_pos_preds) + [0] * len(_neg_preds)
        result = ruleSentimentClassifier.mynumbers(_pos_preds+_neg_preds, y)
        print ("-" * 10 + " Summary " + "-" * 10)
        print ("Classification results:")
        print ("F1: {:.2f}, Recall: {:.2f}, Precision: {:.2f}, Accuracy: {:.2f}".format(
        result[2], result[0], result[1], result[-1]))
        print ("Data distribution: {}".format(dict(Counter(y))))
        print ("-" * 29)


class bowSentimentClassifier(SentimentClassifier):
    def __init__(self, pos_sents, neg_sents, *args):
        super(bowSentimentClassifier, self).__init__(pos_sents, neg_sents)
        self._compute_feature()

    def run(self, test_pos_sents, test_neg_sents):
        test_pos_X = map(self._feature_lookup,
                         bowSentimentClassifier.readlines(test_pos_sents))
        test_neg_X = map(self._feature_lookup,
                         bowSentimentClassifier.readlines(test_neg_sents))
        testy = [1] * len(test_pos_X) + [-1] * len(test_neg_X)
        self.svm = svmclsbinary(name="BOW",
                                X=self.pos_X+self.neg_X,
                                y=self.y,
                                testX=test_pos_X+test_neg_X,
                                testy=testy,
                                docv=False)
        f1, p, r, acc = self.svm._evaluate()
        print ("-" * 10 + " Summary " + "-" * 10)
        print ("Classification results:")
        print ("F1: {:.2f}, Recall: {:.2f}, Precision: {:.2f}, Accuracy: {:.2f}".format(
        f1, r, p, acc))
        print ("Data distribution: {}".format(dict(Counter(self.svm.testy))))
        print ("-" * 29)


    def _compute_feature(self):
        corpus = ""
        for s in self.pos_sents+self.neg_sents: corpus += s.lower()
        vocab = [k for k, v in dict(Counter(corpus.split(" "))).iteritems() if v > 5]
        self.vocab = {k: idx for idx, k in enumerate(vocab)}
        self.vocab_size = len(self.vocab)
        print ("[INFO]: finish computing vocab, start computing features ...")
        self.pos_X = map(self._feature_lookup, self.pos_sents)
        self.neg_X = map(self._feature_lookup, self.neg_sents)
        self.y = [1] * len(self.pos_X) + [-1] * len(self.neg_X)
        print ("[INFO]: finish computing features ... Ready to train SVM ...")

    def _feature_lookup(self, sentence):
        X = []
        words = sentence.split(" ")
        for w in words:
            if w not in self.vocab: continue
            vec = [0.] * self.vocab_size
            vec[self.vocab[w]] = 1.
            X.append(vec)
        if len(X) > 1:
            X = np.sum(X, axis=0)
            assert len(X) == self.vocab_size
            return X
        elif len(X) == 1:
            assert len(X[0]) == self.vocab_size
            return X[0]
        elif len(X) == 0:
            return [0.] * self.vocab_size


class vecSentimentClassifier(bowSentimentClassifier):
    def __init__(self, pos_sents, neg_sents, emb_path):
        self.word2vec = word2vec(emb_path)
        super(vecSentimentClassifier, self).__init__(pos_sents, neg_sents)

    @overrides
    def _compute_feature(self):
        assert self.word2vec is not None
        self.pos_X = map(self._feature_lookup, self.pos_sents)
        self.neg_X = map(self._feature_lookup, self.neg_sents)
        self.y = [1] * len(self.pos_X) + [-1] * len(self.neg_X)
        print ("[INFO]: finish computing features ... Ready to train SVM ...")

    @overrides
    def _feature_lookup(self, sentence):
        X = []
        words = sentence.split(" ")
        for w in words:
            if w not in self.word2vec: continue
            X.append(self.word2vec[w])
        if len(X) > 1:
            return np.sum(X, axis=0)
        elif len(X) == 1:
            return X[0]
        elif len(X) == 0:
            return [0.] * 200
