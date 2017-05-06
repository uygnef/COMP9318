import helper
import operator
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.naive_bayes import MultinomialNB

dic = {}

def position_process(line):
    result = {}
    train_data = {}
    temp_result =None
    for i in line:
        data = i.split(':')[1]
        data = data.split(" ")

        for i, phon in enumerate(data):
            mode_phon = phon
            if phon[-1] in '012':
                mode_phon = phon[:-1]
                if phon[-1] == '1':
                    temp_result = i
            temp = [dic[mode_phon], i]
            if len(data) not in train_data:
                train_data[len(data)] = []
            if len(data) not in result:
                result[len(data)] = []
            result[len(data)].append(temp_result)
            train_data[len(data)].append(temp)
    return train_data, result

def pre_process(line):
    dictionary = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
                  'P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'R', 'S',
                  'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
    for i, word in enumerate(dictionary):
        dic[word] = i
    visual = {}
    # pre = {}
    # next = {}
    #
    # all_pre = {}
    # all_next = {}
    pre_stress = {}
    this_stress = {}
    next_stress = {}

    all_pre_stress = {}
    all_next_stress = {}
    all_stress = {}
    for i in "012":
        visual[i] = {}
    for i in line:
        data = i.split(":")[1]

        ALL_DATA = data.split(" ")

        #######################
        for i, syllable in enumerate(ALL_DATA):
            all_stress = add_syllable(all_stress, syllable)
            temp = syllable
            if syllable[-1] == '1' and i > 4:
                break
            if syllable[-1] in '012':
                temp = temp[:-1]
            if i != 0:
                all_pre_stress = add_syllable(all_pre_stress, temp, ALL_DATA[i-1])
            if i < len(ALL_DATA) - 1:
                all_next_stress = add_syllable(all_next_stress, temp, ALL_DATA[i+1])
            if syllable[-1] == '1':
                if i != 0:
                    pre_stress = add_syllable(pre_stress, syllable[:-1], ALL_DATA[i-1])
                this_stress = add_syllable(this_stress, syllable[:-1])
                if i < len(ALL_DATA) - 1:
                    next_stress = add_syllable(next_stress, syllable[:-1], ALL_DATA[i+1])
    return pre_stress, this_stress, next_stress, all_pre_stress, all_next_stress, all_stress


        #################################
        # for i, phons in enumerate(ALL_DATA):
        #     if(i>0):
        #         if ALL_DATA[i-1][-1] not in '012':
        #             temp2 = ALL_DATA[i-1]
        #         else:
        #             temp2 = ALL_DATA[i-1][:-1]
        #         if phons[-1] not in '012':
        #             new_phon = phons
        #         else:
        #             new_phon = phons[:-1]
        #         if new_phon not in all_pre:
        #             all_pre[new_phon] = {}
        #             all_pre[new_phon][temp2] = 1
        #         elif temp2 not in all_pre[new_phon]:
        #             all_pre[new_phon][temp2] = 1
        #         else:
        #             all_pre[new_phon][temp2] += 1
        #
        #
        #     if(phons[-1] in ['0', '1', '2']):
        #         if phons[:-1] not in visual[phons[-1]]:
        #             visual[phons[-1]][phons[:-1]] = 0
        #         else:
        #             visual[phons[-1]][phons[:-1]] += 1
        #         if(i != 0):
        #             temp1 = ALL_DATA[i-1]
        #             if temp1[-1] in '012':
        #                 temp1 = temp1[:-1]
        #             if phons[:-1] not in pre:
        #                 pre[phons[:-1]] = {}
        #                 pre[phons[:-1]][temp1] = 1
        #             elif temp1 not in pre[phons[:-1]]:
        #                 pre[phons[:-1]][temp1] = 1
        #             else:
        #                 pre[phons[:-1]][temp1] += 1
        #
        #         if(i != len(ALL_DATA) - 1):
        #             temp1 = ALL_DATA[i+1]
        #             if temp1[-1] in '012':
        #                 temp1 = temp1[:-1]
        #             if phons[:-1] not  in next:
        #                 next[phons[:-1]] ={}
        #                 next[phons[:-1]][temp1] = 1
        #             elif temp1 not in next[phons[:-1]]:
        #                 next[phons[:-1]][temp1] = 1
        #             else:
        #                 next[phons[:-1]][temp1] += 1
        #
        #         if(phons[-1] == '1'):
        #             phons = phons[:-1]
        #             last_element = (int(i), phons)
        #         else:
        #             phons = phons[:-1]
        #     temp.append(phons)
        # if(last_element):
        #     temp.append(last_element)
        #     result.append(temp)
    # return result, visual, pre, next, all_pre

def add_syllable(set, index, data=-1):
    if data == -1:
        if index[-1] in '012':
            index = index[:-1]
        if index not in set:
            set[index] = 1
        else:
            set[index] += 1
        return set
    if data[-1] in '012':
        data = data[:-1]
    if index not in '012':
        if index not in set:
            set[index] = {}
            set[index][data] = 1
        elif data not in set[index]:
            set[index][data] = 1
        else:
            set[index][data] += 1
    else:
        if index not in set:
            set[index] = {}
            set[index][data] = 1
        else:
            set[index][data] += 1
    return set

def draw_bar(visual):
    D = sorted(visual.items(), key=operator.itemgetter(1), reverse=True)
    performance = [i[1] for i in D]
    D_x = [i[0] for i in D]
    y_pos = np.arange(len(D_x))
    return D_x, y_pos, performance

# def draw(list, list2):
#     params = {'legend.fontsize': 20,
#               'figure.figsize': (20, 5),
#               'axes.labelsize': 20,
#               'axes.titlesize': 20,
#               'xtick.labelsize': 12,
#               'ytick.labelsize': 20}
#     plt.rcParams.update(params)
#     D_x, y_pos, performance = draw_bar(list)
#     y = []
#     for i in D_x:
#         y.append(list2[i])
#     plt.bar(np.arange(len(D_x)), y)
#     plt.bar(np.arange(len(D_x)), performance)
#     plt.xticks(y_pos, D_x)
#     plt.ylabel('Times')
#     plt.show()



def all_draw(list, list2):
    params = {'legend.fontsize': 20,
              'figure.figsize': (20, 5),
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'xtick.labelsize': 12,
              'ytick.labelsize': 20}
    plt.rcParams.update(params)
    for j, i in enumerate(sorted(list)):
        a = plt.subplot(3, 3, (j)%9 + 1)
        y = []

        D_x, y_pos, performance = draw_bar(list[i])
        for p in D_x:
            y.append(list2[i][p])

        plt.bar( np.arange(len(D_x)), y)
        plt.bar(np.arange(len(D_x)), performance)
        print(y)
        print(performance)
      #  plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, D_x)
        plt.ylabel('Times')
        plt.text(0.9,0.9 ,"Next "+ str(i),fontsize = 20, ha='center', va='center',transform=a.transAxes)

        if (j)%9 == 8:
            plt.show()
    plt.show()

raw_data = helper.read_data("./asset/training_data.txt")
pre_stress, this_stress, next_stress, all_pre_stress, all_next_stress, all_stress = pre_process(raw_data)
my_all_pre = {}
for i in pre_stress:
    my_all_pre[i] = all_pre_stress[i]




train, result = position_process(raw_data)
print(train)
print(result)
# train_data = train[:int(len(train)*0.8)]
# test_data = train[int(len(train)*0.8):]
# train_result = result[:int(len(train)*0.8)]
# test_result = result[int(len(train)*0.8):]

def divide_train_test(data, result):
    train = []
    test = []
    train_result = []
    test_result = []
    for i, value in enumerate(data):
        if(random.randint(1)>0.1):
            train.append(value)
            train_result.append(result[i])
        else:
            test.append(value)
            test_result.append(result[i])
    return train, train_result, test, test_result



clf = [MultinomialNB() for _ in range(len(train))]
all_test_data = {}
all_test_result = {}
print(train)
for i,value in enumerate(train):
    train_data, train_result, test_data, test_result = divide_train_test(value, result[i])
    clf[i].fit(train_data[i], train_result)

exit(2)

def most_prob(data, clf):
    prob = 0
    result = None
    temp_prob = clf.predict_proba(data)
    print(temp_prob)
    for i,value in enumerate(data):
        break
        print(value)
        temp_prob = clf.predict_proba(value)
        print(temp_prob)
        if temp_prob > prob:
            result = i
            prob = temp_prob
    return result

i = 0
j = 0
length = test_data[0][-1]
next_data = False
temp = []
pi = 0
for i, value in enumerate(test_data):
    if next_data:
        length = value[-1]
        temp = []
        j = 0
        next_data = False
    j += 1
    temp.append(value)
    if (j == length):
        next_data = True
        result = most_prob(temp,clf)
        pi = i - length + 1
        if(result != test_result[i]):
            print(1)
        else:
            print(0)

# for i in visual:
#     print(visual[i])
#sorted_x = sorted(visual['1'].items(), key=operator.itemgetter(1))
#print(sorted_x)



# D  = sorted(visual['1'].items(), key=operator.itemgetter(1), reverse=True)
# performance = [i[1] for i in D]
# D_x = [i[0] for i in D]
# y_pos = np.arange(len(D_x))


# all_draw(pre, all_pre)
