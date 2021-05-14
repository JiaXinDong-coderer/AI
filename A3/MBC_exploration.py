import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB


def readData(path):
    """
    This function use os to traverse around the root folder for train
    and evaluation data and put them into x and y set
    no nparray is used in this function

    :param path: A root path of both train and evaluation data
    :return: x set and y set
    """
    x = []
    y = []
    body = re.compile(r'.*Lines:.*')
    dirLst = [os.path.join(path,i) for i in os.listdir(path)]
    for i in dirLst:
        for root, dirs, files in os.walk(i):
            for file in files:
                file_path = os.path.join(root, file)
                f = open(file_path,'r',encoding='utf-8-sig',errors='ignore')
                # split header and body
                result = body.split("".join(f.readlines()))
                result[-1] = result[-1].replace(">","").replace("<","") # get rid of > <
                x.append(result[-1]) # get whole passage in one
                y.append(dirLst.index(i)) # encode y into 0,1,2,3
                f.close()
    return x,y

def getMultiHot(lst):
    """
    use Count Vectorizer to extract multihot encoding of each set x
    :param lst: should be x dta set
    :return: a transformed x and a lexica contain all x's vocabs
    """
    vc = CountVectorizer(ngram_range=(1,1))
    x = vc.fit_transform(lst)
    return x.toarray(), vc.get_feature_names()

def learningCurve(model,x,y,evaX,evaY):
    """
    this is a self implementing LC function. it will split data into each different percent groups
    (20,40,60,80(%)) and use each partial x to train model to see it's percision and recall
    :param model: a model to train data
    :param x: x train
    :param y: y train
    :param evaX: x evaluation
    :param evaY: y evaluation
    :return: a list contain all f scores of (20,40,60,80(%))
    """
    f1 = []
    # split 4 times
    x1, X_test, y1, y_test = train_test_split(x, y, test_size=0.8)
    model.fit(x1,y1)
    ypred = model.predict(evaX)
    f1.append(f1_score(ypred,evaY,average="macro"))
    x1, X_test, y1, y_test = train_test_split(x, y, test_size=0.6)
    model.fit(x1, y1)
    ypred = model.predict(evaX)
    f1.append(f1_score(ypred, evaY,average="macro"))
    x1, X_test, y1, y_test = train_test_split(x, y, test_size=0.4)
    model.fit(x1, y1)
    ypred = model.predict(evaX)
    f1.append(f1_score(ypred, evaY,average="macro"))
    x1, X_test, y1, y_test = train_test_split(x, y, test_size=0.2)
    model.fit(x1, y1)
    ypred = model.predict(evaX)
    f1.append(f1_score(ypred, evaY,average="macro"))

    return f1

def LRCV(x,y,evaX,evaY):
    """
    This is LR model
    :param x: x train
    :param y: y train
    :param evaX: x evaluation
    :param evaY: y evaluation
    :param encoding: UB or BB
    :return: list of f score if LC is 1 and unigram encoding
    """
    LR = LogisticRegression(max_iter=200).fit(x, y)
    precitY = LR.predict(evaX)

    precisonRecallF = metrics.precision_recall_fscore_support(evaY, precitY, average="macro")
    print(precisonRecallF)
    print("LR,{0},{1:.2f},{2:.2f},{3:.2f}".format("UBSW", precisonRecallF[0], precisonRecallF[1], precisonRecallF[2]))
    f.write(
        "LR,{0},{1:.2f},{2:.2f},{3:.2f}\n".format("UBSW", precisonRecallF[0], precisonRecallF[1], precisonRecallF[2]))

    if LC == "1":
        f1 = learningCurve(LR,x,y,evaX,evaY)
        return f1
    return None

def NB(x,y,evaX,evaY):
    """
        This is NB model
        :param x: x train
        :param y: y train
        :param evaX: x evaluation
        :param evaY: y evaluation
        :param encoding: UB or BB
        :return: list of f score if LC is 1 and unigram encoding
        """

    NBmodel = BernoulliNB(alpha=0.0001)
    NBmodel.fit(x,y)
    predictY = NBmodel.predict(evaX)

    precisonRecallF = metrics.precision_recall_fscore_support(evaY,predictY,average="macro")
    print(precisonRecallF)
    print("NB,{0},{1:.2f},{2:.2f},{3:.2f}".format("UBSW",precisonRecallF[0],precisonRecallF[1],precisonRecallF[2]))
    f.write("NB,{0},{1:.2f},{2:.2f},{3:.2f}\n".format("UBSW",precisonRecallF[0],precisonRecallF[1],precisonRecallF[2]))
    #accracy: ((evaY.shape[0]-(evaY != predictY).sum())/predictY.shape[0])
    if LC == "1":
        f1 = learningCurve(NBmodel,x,y,evaX,evaY)
        return f1
    return None

def SVM(x,y,evaX,evaY):
    """
        This is SVC model
        :param x: x train
        :param y: y train
        :param evaX: x evaluation
        :param evaY: y evaluation
        :param encoding: UB or BB
        :return: list of f score if LC is 1 and unigram encoding
        """
    SVC = make_pipeline(StandardScaler(), SGDClassifier())
    SVC.fit(x, y)
    predictY = SVC.predict(evaX)
    precisonRecallF = metrics.precision_recall_fscore_support(evaY, predictY, average="macro")
    print(precisonRecallF)
    print("SVM,{0},{1:.2f},{2:.2f},{3:.2f}".format("UBSW", precisonRecallF[0], precisonRecallF[1], precisonRecallF[2]))
    f.write(
        "SVM,{0},{1:.2f},{2:.2f},{3:.2f}\n".format("UBSW", precisonRecallF[0], precisonRecallF[1], precisonRecallF[2]))
    if LC == "1":
        f1 = learningCurve(SVC,x,y,evaX,evaY)
        return f1
    return None

def RF(x,y,evaX,evaY):
    """
        This is RF model
        :param x: x train
        :param y: y train
        :param evaX: x evaluation
        :param evaY: y evaluation
        :param encoding: UB or BB
        :return: list of f score if LC is 1 and unigram encoding
        """
    forest = RandomForestClassifier(max_depth=6, random_state=0)
    forest.fit(x,y)
    predictY = forest.predict(evaX)
    precisonRecallF = metrics.precision_recall_fscore_support(evaY, predictY, average="macro")
    print(precisonRecallF)
    print("RF,{0},{1:.2f},{2:.2f},{3:.2f}".format("UBSW", precisonRecallF[0], precisonRecallF[1], precisonRecallF[2]))
    f.write(
        "RF,{0},{1:.2f},{2:.2f},{3:.2f}\n".format("UBSW", precisonRecallF[0], precisonRecallF[1], precisonRecallF[2]))
    if LC == "1":
        f1 = learningCurve(forest,x,y,evaX,evaY)
        return f1
    return None

# python MBC_exploration.py <trainset> <evalset> <output> <display LC>

# python MBC_exploration.py trainset evalset output displayLC
# python MBC_exploration.py Selected_20NewsGroup\Training Selected_20NewsGroup\evaluation outputMBC
# python MBC_exploration.py Selected_20NewsGroup\Training Selected_20NewsGroup\evaluation outputMBC
# python MBC_exploration.py Selected_20NewsGroup\evaluation Selected_20NewsGroup\Training outputMBCgg


if __name__ == '__main__':
    if len(argv) != 4:
        print("Usage: python MBC_exploration.py <trainset> <evalset> <output>")
        exit(-1)

    trainPath = argv[1]
    evaPath = argv[2]
    outputPath = argv[3]
    global LC
    LC = '0'

    # read data
    BBtrainX, BBtrainY = readData(trainPath)
    BBevaX, BBevaY = readData(evaPath)

    # Here we use TfidfVectorizer to extract different features
    vc1 = TfidfVectorizer(stop_words='english',ngram_range=(1,1)) # extract features here
    x1 = vc1.fit_transform(BBtrainX)
    BBtrainX = x1.toarray()
    BBlexica = vc1.get_feature_names()

    BBtrainY = np.array(BBtrainY)
    vc2 = CountVectorizer(stop_words='english',ngram_range=(1,1),vocabulary=np.array(BBlexica))
    x2 = vc2.fit_transform(BBevaX)
    BBevaX = x2.toarray()
    BBevaY = np.array(BBevaY)

    print(BBtrainX.shape)
    print(BBevaX.shape)

    global f
    f = open(outputPath,'w')

    f2 = NB(BBtrainX, BBtrainY, BBevaX, BBevaY)

    f1 = LRCV(BBtrainX, BBtrainY, BBevaX, BBevaY)

    f3 = SVM(BBtrainX, BBtrainY, BBevaX, BBevaY)

    f4 = RF(BBtrainX, BBtrainY, BBevaX, BBevaY)

    f.close()

    # make plot if LC = 1
    if LC == "1":
        fig = plt.figure(figsize=(10, 6))
        l1 = plt.plot(np.array([0.2, 0.4, 0.6, 0.8]), f1, color='red', label="LR")
        l2 = plt.plot(np.array([0.2, 0.4, 0.6, 0.8]), f2, color='blue', label="NB")
        l3 = plt.plot(np.array([0.2, 0.4, 0.6, 0.8]), f3, color='yellow', label="SVM")
        l4 = plt.plot(np.array([0.2, 0.4, 0.6, 0.8]), f4, color='green', label="RF")
        plt.legend()
        plt.title('Learning Curve')
        plt.xlabel('percent of data')
        plt.ylabel('f-score')
        plt.show()
        print(f1)
        print(f2)
        print(f3)
        print(f4)
