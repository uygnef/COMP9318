import operator
from collections import Counter
################# Question 1 #################

def multinomial_nb(training_data, sms):# do not change the heading of the function
    data_set = pre_process(training_data)
    result = 1
    for word in sms:
        ham_occur = occur(word, data_set['ham'])
        ham_posib = ham_occur / ( smooth_len(sms, data_set) + sum(data_set['ham'].values()) )
        spam_occur = occur(word, data_set['spam'])
        spam_posib = spam_occur / ( smooth_len(sms, data_set)+ sum(data_set['spam'].values()) )
        print(word, spam_occur, spam_posib, "...", ham_occur, ham_posib)
        result = result * (spam_posib/ham_posib)
        print("reslut", result)
    return result


def smooth_len(sms, dataset):

    return len( set().union(sms, list(dataset['spam'].keys()), list(dataset['ham'].keys())) )

def occur(word, data_set):
    if word not in data_set:
        return 1
    return 1 + data_set[word]


def merge_dicts(b , a):
    A = Counter(a)
    B = Counter(b)
    C = A+B
    return C


def pre_process(training_data): #return all words in the same tag
    result = {}
    for i in training_data:
        if i[1] not in result:
            result[i[1]] = i[0]
        else:
            result[i[1]] = merge_dicts(result[i[1]], i[0])
    return result




