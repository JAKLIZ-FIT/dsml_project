# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:19:30 2023

@author: jiri zilka
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

nanColumns = ['ionizationclass','FluxCompensation','multideminsionality',
               'error','error_type']

def loadData(path='../data/manufacturing.csv'):
    return pd.read_csv(path)

def prepareData(df,encoding='basic'):
    df.drop("id", axis = 1,inplace=True)
    if encoding == 'basic':
        le = LabelEncoder()
        for column in nanColumns:
            df[column] = le.fit_transform(df[column])
    elif encoding == 'onehot':
        pass # TODO
    else:
        pass
    return df
    
def removeOutliers(df):
    #TODO
    pass

def normalize(df):
    #TODO
    pass

if __name__ == '__main__':
    df = loadData()
    df = prepareData(df)
    