import helper
import pandas as pd

class WORD():
    def __init__(self, data):
        lines = [i.split(":") for i in data]
        self.df = pd.DataFrame(data=lines, columns=("word", "all phoneme"))
