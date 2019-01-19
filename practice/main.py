from Classifier import *


if __name__ == "__main__":
    train_pos_sents = "./dataset/train_pos.txt"
    train_neg_sents = "./dataset/train_neg.txt"

    test_pos_sents = "./dataset/test_pos.txt"
    test_neg_sents = "./dataset/test_neg.txt"

    emb_path = "./myemb.vec"

    print ("\n")
    print ("[INFO]: running rule-based experiment ...")
    mycls = ruleSentimentClassifier(test_pos_sents, test_neg_sents)
    print ("\n")

    print ("[INFO]: running BOW experiment with one-hot feature ...")
    mycls = bowSentimentClassifier(train_pos_sents, train_neg_sents)
    mycls.run(test_pos_sents, test_neg_sents)
    print ("\n")


    print ("[INFO]: running BOW experiment with word2vec feature ...")
    mycls = vecSentimentClassifier(train_pos_sents, train_neg_sents, emb_path)
    mycls.run(test_pos_sents, test_neg_sents)
    print ("\n")
