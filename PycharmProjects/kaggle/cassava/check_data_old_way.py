import pandas as pd
import zipfile

from subprocess import check_output

def ck():
    print(check_output(["ls", "./data/"]).decode("utf8"))
    Dataset = 'cassava-disease'

    with zipfile.ZipFile('./data/' + Dataset + ".zip", "r") as z:
        z.extractall("./data")

def ck_data():

    train = 'train'
    test = 'test'

    with zipfile.ZipFile('./data/' + train + ".zip", "r") as z:
        z.extractall("./data/")

    with zipfile.ZipFile('./data/' + test + ".zip", "r") as z:
        z.extractall("./data/")

if __name__ == '__main__':
    ck_data()