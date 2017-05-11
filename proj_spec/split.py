import helper
import random

def split():
    line = helper.read_data("./asset/training_data.txt")

    train_file = open("./asset/train.txt", 'w+')
    test_file = open("./asset/test.txt", 'w+')
    for i in line:
        if random.randint(0,9):
            train_file.write(i+'\n')
        else:
            test_file.write(i+'\n')

split()