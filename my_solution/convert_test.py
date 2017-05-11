import helper
import pickle

def convert():
    raw_data = helper.read_data('./asset/test.txt')
    result = []
    file = open("result.txt", "w+b")
    for i in raw_data:
        counter = 0
        for char in i:
            if char in '012':
                counter += 1
                if char == '1':
                    result.append(counter)
                    break
    pickle.dump(result,file)

def get_result():
    result = pickle.load(open("result.txt", "rb"))
    return result



