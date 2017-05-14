import helper
import scipy

class word:
    def __init__(self, str):
        self.name, self.phonem = str.split(':')
        self.syllables = []
        for i in self.phonem.split(" "):
            self.syllables.append(syllbale(i))
        self.len = 0
        self.result = 0
        temp = 0
        for i in self.syllables:
            if i.is_vowel:
                self.len += 1
                if i.stress == 1:
                    self.result = temp + 1
                temp += 1

        self.pho_len = len(self.syllables)

    def __repr__(self):
        string = self.name + ":" + str(self.pho_len) +":"
        for i in self.syllables:
            string += i.name + " "
            if i.is_vowel:
                string += str(i.stress) + " "
        return string + ":"+str(self.result)+"/"+str(self.len)


class syllbale:
    def __init__(self, str):
        self.is_vowel = False
        self.stress = None
        self.name = str
        if str[-1] in '012':
            self.is_vowel = True
            self.stress = int(str[-1])
            self.name = str[:-1]


def pre_pos():
    training_data = helper.read_data('../asset/test_words.txt')
    a = []
    for i in training_data:
        temp = word(i)
        a.append(temp)
    return a

a = pre_pos()


volum = [[], [], [],[]]
for i in a:
    print(i.len-1)
    volum[i.len-1].append(i)
s = 0
for i in volum:
    s += 1
    print(scipy.stats.pearsonr([j.phon_len for j in volum], [j.result for j in volum]))