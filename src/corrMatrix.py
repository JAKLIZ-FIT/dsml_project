import pandas as pd
from sklearn.preprocessing import LabelEncoder

from dataPrep import *

def showCorrMatrix(df):
    corrMatrix = df.corr()
    print(corrMatrix)
    return corrMatrix

if __name__ == '__main__':
    df = loadData()
    print(df)
    df = prepareData(df)
    cm = showCorrMatrix(df)