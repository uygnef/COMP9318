from submission import *
import convert_test
import helper
import nltk



training_data = helper.read_data('./asset/test_words.txt')
classifier_path = './asset/classifier.dat'

train(training_data, classifier_path)

test_data = helper.read_data('./asset/t est.txt')
# for i in training_data:
#     a = i.split(":")[0]
#     if(nltk.pos_tag([a])[0][-1] not in ['NN', 'NNP', 'NNS']):
#         print(a, nltk.pos_tag([a]))


prediction = test(test_data, classifier_path)
result = convert_test.get_result()
# print(result)
# print(prediction)
# a = {}
# a[1] = 0
# a[2] = 0
# a[3] = 0
# a[4] = 0
# la = [0,0 , 0, 0, 0]
# for i in range(len(result)):
#     if result[i] == 4:
#         print(test_data[i])
#     if result[i] != prediction[i]:
#         a[result[i]] += 1
#     la[result[i]] += 1
# for i in a:
#     print(i, '\t', a[i]/la[i], '\t', la[i])
from sklearn.metrics import f1_score
print(f1_score(result, prediction, average='macro'))