import numpy as np
from sys import argv
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def readData(path):
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
    vc = CountVectorizer(ngram_range=(1,1))
    x = vc.fit_transform(lst)
    return x.toarray(), vc.get_feature_names()

def learningCurve(model,x,y,evaX,evaY):
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


# python MBC_final.py <trainset> <evalset> <output> <display LC>

# python MBC_final.py trainset evalset output displayLC
# python MBC_final.py Selected_20NewsGroup\Training Selected_20NewsGroup\evaluation outputMBCF
# python MBC_final.py Selected_20NewsGroup\evaluation Selected_20NewsGroup\Training outputMBCF


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

    # do bi-pair (BB)
    vc1 = TfidfVectorizer(stop_words='english',ngram_range=(1,2))
    x1 = vc1.fit_transform(BBtrainX)
    BBtrainX = x1.toarray()
    BBlexica = vc1.get_feature_names()

    BBtrainY = np.array(BBtrainY)
    vc2 = CountVectorizer(stop_words='english',ngram_range=(1,2),vocabulary=np.array(BBlexica))
    x2 = vc2.fit_transform(BBevaX)
    BBevaX = x2.toarray()
    BBevaY = np.array(BBevaY)

    print(BBtrainX.shape)
    print(BBevaX.shape)

    global f
    f = open(outputPath,'w')

    f1 = LRCV(BBtrainX, BBtrainY, BBevaX, BBevaY)

    f.close()