import helper
from sklearn import tree
from sklearn.svm import SVC
import pickle
from nltk.stem.lancaster import LancasterStemmer

def train(data, classifier_file):
    train_data, index_file = pre_process(data)
    clf = {}
    for i in [2, 3, 4]:
        temp_train = [j[:-1] for j in train_data[i]]
        temp_result = [j[-1] for j in train_data[i]]
        # if i in [3,4]:
        #     clf[i] = SVC(probability = True)
        # else:
        class_weight = {}
        class_weight[0] = 1
        class_weight[1] = 5
        clf[i] = tree.DecisionTreeClassifier(class_weight=class_weight, max_depth=18)
        clf[i].fit(temp_train, temp_result)
    file = open(classifier_file, 'wb')
    pickle.dump((clf, index_file), file)


def test(data, classifier_file):
    file = open(classifier_file, 'rb')
    clf, index_file = pickle.load(file)
    test_data = get_data(data, index_file)
    result = []
    for word in test_data:
        index = len(word)
        predict = []
        for syllable in word:
            predict.append(clf[index].predict_proba([syllable])[0][-1])
        # if len(predict) == 4:
        #     predict[3] *= 30
        #     predict[2] *= 4
        # if len(predict) == 3:
        #     predict[2] *= 5
        pred = predict.index(max(predict))
        result.append(pred + 1)
    return result


def add_dict(index, dict):
    if index not in dict:
        dict[index] = 1
    else:
        dict[index] += 1


def clear_phon(phone):
    if phone[-1] in '012':
        return phone[:-1]
    return phone


def pre_process(line):
    # dict = {}
    # dictionary = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
    #               'P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'R', 'S',
    #               'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
    # # pre None = 39 next none = 40
    # # tags = {'NN':1, 'NNP':2, 'NNS':3, 'JJ':4, 'RB':5, 'IN':6, 'DT':7, 'JJR':8, 'VB':9}
    # for i, word in enumerate(dictionary):
    #     dict[word] = i

    train_data = {}
    stress_occ = {}
    total_occ = {}
    pre_occ = {}
    next_occ = {}

    prefix = {}
    suffix = {}
    for i in line:
        phons = i.split(':')[1].split(" ")
        word = i.split(':')[0]
        pre = 'PRE'
        add_dict('PRE', total_occ)
        add_dict('NEXT', total_occ)
        pref, suff = get_pre_suff(word)
        add_dict(pref, prefix)
        add_dict(suff, suffix)
        for index, j in enumerate(phons):
            if j[-1] in '012':
                if j[-1] == '1':
                    add_dict(j[:-1], stress_occ)
                    add_dict(pre, pre_occ)
                    if index == len(phons) - 1:
                        add_dict('NEXT', next_occ)
                    else:
                        add_dict(clear_phon(phons[index + 1]), next_occ)
                j = j[:-1]
            add_dict(j, total_occ)
            pre = j

    # feature calibration
    for i in stress_occ:
        stress_occ[i] = stress_occ[i] / total_occ[i]
    #    print(i,end=' ')
    #print()
    for i in pre_occ:
        pre_occ[i] = pre_occ[i] / total_occ[i]
    #    print(i,end=' ')
   # print()

    for i in next_occ:
        next_occ[i] = next_occ[i] / total_occ[i]
    #   print(i,end=' ')


    for i in line:
        # a = nltk.pos_tag([i.split(':')[0]])[0][-1]
        # tag = tags.get(a, 0)
        data = i.split(':')[1].split(" ")
        position = 0
        temp_list = []
        word = i.split(':')[0]
        pref, suff = get_pre_suff(word)
        for i, phon in enumerate(data):
            if phon[-1] in '012':
                temp_result = 0
                if phon[-1] == '1':
                    temp_result = 1
                phon = phon[:-1]
                if i == 0:
                    pre = 'PRE'
                elif data[i-1] in  ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']:
                    pre = 'PRE'
                else:
                    pre = clear_phon(data[i - 1])

                if i == len(data) - 1:
                    next = 'NEXT'
                elif data[i+1] in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']:
                    next = 'NEXT'
                else:
                    next = clear_phon(data[i + 1])
                if pre not in pre_occ:
                    pre_occ[pre] = 0
                if next not in next_occ:
                    next_occ[next] = 0
                if phon not in stress_occ:
                    stress_occ[phon] = 0
                    #               Phoneme name   No of vowel  pre Phoneme   next Phoneme                result
                temp_list.append([stress_occ[phon], position, pre_occ[pre], next_occ[next], prefix[pref], suffix[suff], temp_result])
                position += 1

        if len(temp_list) not in train_data:
            train_data[len(temp_list)] = []
        train_data[len(temp_list)] += temp_list
    index_list = [stress_occ, pre_occ, next_occ, prefix, suffix]
    return train_data, index_list

def get_pre_suff(word):
    st = LancasterStemmer()
    new_word = st.stem(word)
    a = (word.lower()).split(new_word)
    if(len(a) < 2):
        a = ['', '']
    return a[0], a[1]


def get_data(line, index_list):
    # dict = {}
    # dictionary = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
    #               'P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'R', 'S',
    #               'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
    # # tag_list = {'NN': 1, 'NNP': 2, 'NNS': 3, 'JJ': 4, 'RB': 5, 'IN': 6, 'DT': 7, 'JJR': 8, 'VB': 9}
    # for i, word in enumerate(dictionary):
    #     dict[word] = i

    train_data = []
    stress_occ = index_list[0]
    pre_occ = index_list[1]
    next_occ = index_list[2]
    prefix = index_list[3]
    suffix = index_list[4]
    for i in line:
        # a = nltk.pos_tag([i.split(':')[0]])[0][-1]
        # tag = tag_list.get(a, 0)

        data = i.split(':')[1].split(" ")
        pref, suf = get_pre_suff(i.split(':')[0])
        if(pref not in prefix):
            pref = ''
        if(suf not in suffix):
            suf = ''
        temp_list = []
        position = 0

        total = 0
        for i, phon in enumerate(data):
            if phon in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']:
                total += 1
        for i, phon in enumerate(data):
            if phon in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']:
                if i == 0:
                    pre = pre_occ['PRE']
                else:
                    pre = pre_occ[data[i - 1]]
                if i == len(data) - 1:
                    next = next_occ['NEXT']
                else:
                    next = next_occ[data[i + 1]]

                temp_list.append([stress_occ[phon], position, pre, next, prefix[pref], suffix[suf]])
                position += 1
        train_data.append(temp_list)
    return train_data
