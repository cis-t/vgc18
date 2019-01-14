from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score


class svmforword(object):
    def __init__(self, name, eval_logger, y, testy, module):
        self.name = name
        self.eval_logger = eval_logger
        self.y_dist = Counter(y)
        self.testy_dist = Counter(testy)
        self.module = module


class svmclsbinary(object):
    def __init__(self, name, X, y, testX, testy, kernel="linear", C=1, degree=1, docv=True):
        self.name = name
        self.X = X
        self.y = y
        self.testX = testX
        self.testy = testy
        self.train_finished = False
        self.eval_finished = False
        if docv:
            parastotune = [{"kernel": ["linear"],
                            "C": [1, 10, 100],
                            "degree": [2, 3, 4]}]
            clf = GridSearchCV(SVC(), parastotune, cv=5, scoring="f1_macro")
            clf.fit(self.X, self.y)
            C = clf.best_params_["C"]
            degree = clf.best_params_["degree"]
            kernel = clf.best_params_["kernel"]
            print ("CV finished ... Using C={}, degree={}, kernel={}".format(
                C, degree, kernel))
            del clf, parastotune
        self.module = svm.SVC(kernel=kernel, C=C, degree=degree, probability=True)
        self._train()

    def _train(self):
        assert self.train_finished == False
        self.module.fit(self.X, self.y)
        self.train_finished = True
        print ("[INFO]: finish training this SVM ...")

    def _evaluate(self):
        assert self.eval_finished == False
        pred = self._predict(self.testX)
        r, p, f1, acc = svmclsbinary.mynumbers(pred, self.testy)
        self.eval_logger = "f1: {}, recall: {}, precision: {}, acc: {}. TestSize::{}, TestDist::{}".format(
            f1, r, p, acc, len(self.testy), Counter(self.testy))
        self.eval_finished = True
        return (f1, p, r, acc)

    def _predict(self, myX):
        assert self.train_finished == True
        return self.module.predict(myX)

    def save_log(self):
        assert self.train_finished == True
        assert self.eval_finished == True
        return svmforword(self.name,
                          self.eval_logger,
                          self.y,
                          self.testy,
                          self.module)

    @staticmethod
    def mynumbers(pred, y):
        recall = recall_score(y, pred)
        precision = precision_score(y, pred)
        f1 = f1_score(y, pred)
        acc = accuracy_score(y, pred)
        return (recall, precision, f1, acc)
