import helper
from sklearn import tree
import pickle


def train(data, classifier_file):
    train_data = pre_process(data)
    clf = {}
    for i in train_data:
        temp_train = [j[:-1] for j in train_data[i]]
        temp_result = [j[-1] for j in train_data[i]]
        clf[i] = tree.DecisionTreeClassifier()
        clf[i].fit(temp_train, temp_result)
    file = open(classifier_file, 'wb')
    pickle.dump(clf, file)


def test(data, classifier_file):
    file = open(classifier_file, 'rb')
    clf = pickle.load(file)
    test_data = get_data(data)
    result = []
    for word in test_data:
        index = len(word)
        predict = []
        for syllable in word:
            predict.append(clf[index].predict_proba([syllable])[0][-1])
        pred = predict.index(max(predict))
        result.append(pred+1)
    return result



def pre_process(line):
    dict = {}
    dictionary = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
                  'P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'R', 'S',
                  'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
    #pre None = 39 next none = 40

    for i, word in enumerate(dictionary):
        dict[word] = i

    train_data = {}
    for i in line:
      #  tag = tag_list[nltk.pos_tag(i.split(':')[0])[0][-1]]
        data = i.split(':')[1].split(" ")
        position = 0
        temp_list = []
        for i, phon in enumerate(data):
            if phon[-1] in '012':
                temp_result = 0
                if phon[-1] == '1':
                    temp_result = 1
                phon = phon[:-1]
                if i == 0:
                    pre = 39
                else:
                    if data[i - 1][-1] in '012':
                        pre = dict[data[i-1][:-1]]
                    else:
                        pre = dict[data[i-1]]
                if i == len(data) - 1:
                    next = 40
                else:
                    if data[i+1][-1] in '012':
                        next = dict[data[i+1][:-1]]
                    else:
                        next = dict[data[i+1]]

                temp_list.append([dict[phon], position, pre, next, temp_result])
                position += 1

        if len(temp_list) not in train_data:
            train_data[len(temp_list)] = []
        train_data[len(temp_list)] += temp_list
    return train_data

def get_data(line):
    dict = {}
    dictionary = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
                  'P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'R', 'S',
                  'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
    tag_list = {}
    for i, word in enumerate(dictionary):
        dict[word] = i


    train_data = []
    for i in line:
    #    tag = tag_list[nltk.pos_tag(i.split(':')[0])[0][-1]]
        data = i.split(':')[1].split(" ")
        temp_list = []
        position= 0
        for i, phon in enumerate(data):
            if phon in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']:
                if i == 0:
                    pre = 39
                else:
                    pre = dict[data[i-1]]
                if i == len(data) - 1:
                    next = 40
                else:
                    next = dict[data[i+1]]

                temp_list.append([dict[phon], position, pre, next])
                position += 1
        train_data.append(temp_list)
    return train_data
training_data = helper.read_data('./asset/training_data.txt')
classifier_path = './asset/classifier.dat'
train(training_data, classifier_path)
test_data = helper.read_data('./asset/tiny_test.txt')
prediction = test(test_data, classifier_path)
print(prediction)
