## import modules here 
import helper
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import random

from sklearn.metrics import f1_score
dic = {}

################# training #################

def train(data, classifier_file):# do not change the heading of the function
    pass # **replace** this line with your code    

################# testing #################

def test(data, classifier_file):# do not change the heading of the function
    pass # **replace** this line with your code


def str_phon_to_int(phons):
    return int(dic[phons])


def pre_process(line):
    dictionary = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
                  'P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'R', 'S',
                  'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
    for i, word in enumerate(dictionary):
        dic[word] = i
    result = []
    visual = {}
    for i in line:
        data = i.split(":")[1]
        temp = []
        for i, phons in enumerate(data.split(" ")):
            if(phons[-1] in ['0', '1', '2']):
                if phons[-1] not in visual:
                    visual[phons[-1]] = [phons[:-1]]
                else:
                    visual[phons[-1]].append(phons[:-1])

                if(phons[-1] == '1'):
                    phons = phons[:-1]
                    last_element = (int(i), phons)
                else:
                    phons = phons[:-1]
            temp.append(str_phon_to_int(phons))
        if(last_element):
            temp.append(last_element)
            result.append(temp)
    return result, visual

def draw(data):
    result = []
    for i in data:
        if i[-1][0] == 0:
            pre = -1
        else:
            pre = i[i[-1][0] -1]
        if i[-1][0] == len(i) - 1:
            suff = -1
        else:
            suff = i[i[-1][0] +1]
        result.append((pre, suff))
    return result

ab = helper.read_data("./asset/training_data.txt")
b, vasual = pre_process(ab)
seprate_point = math.ceil(len(b)*0.8)
tran_data = b[:seprate_point]
test_data = b[seprate_point:]

# visual_data = {}
# for i in tran_data:
#     if i[-1][1] in visual_data:
#         visual_data[i[-1][1]].append(i)
#     else:
#         visual_data[i[-1][1]] = [i]
#
# for i in visual_data:
#     data = draw(i)
#     data_fram = pd.DataFrame(data,index=['pre', 'next'])
#     break
#
# print(data_fram)
result = []
for i in test_data:
    a = len(i[:-1]) - 4
    if a <0:
        a = 0
    result.append(a)


# T_X= []
# for i in test_data:
#     a = i[:-1]
#     while(len(a)<5):
#         a.append(39)
#     if(len(a) > 5):
#         a = a[:5]
#     T_X.append(a)
#
# X = []
# Y = []
# for i in tran_data:
#     X.append(i[:-1])
#     while(len(X[-1]) < 5):
#         X[-1].append(39)
#     if(len(X[-1]) > 5):
#         X[-1] = X[-1][:5]
#     Y.append(i[-1][0])
#
#
# max_d = 0
# for i in X:
#     max_d = max(len(i), max_d)
#
# # neigh = KNeighborsClassifier(n_neighbors=3)
# # neigh.fit(X, Y)
#
# clf = MultinomialNB()
# clf.fit(X, Y)
#
#
# mismatch = 0
# predict = clf.predict(T_X)
real = [tran_data[i][-1][0] for i in range(len(test_data))]
# # for i, data in enumerate(T_X):
# #     predict = neigh.predict([data])[0]
# #     if(predict != tran_data[i][-1][0]):
# #       #  print(predict, tran_data[i][-1])
# #         mismatch += 1
aall = 0
# my_predic = []
for i in range(len(real)):
    if(real[i] != result[i]):
        print(real[i], result[i])
        aall += 1
print(aall/len(real))
#
for i in ab:
    print(i)
print(f1_score(real, result, average="macro"))
#
# #.predict_proba(test_data[0][:-1])
# #print(y_pred, test_data[0][-1])
