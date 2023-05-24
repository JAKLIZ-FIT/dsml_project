# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:19:30 2023

@author: jiri zilka
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, normalize

nanColumns = ['ionizationclass','FluxCompensation','multideminsionality',
               'error','error_type']

def loadData(path='../data/manufacturing.csv'):
    return pd.read_csv(path)

"""
encoding: basic, onehot, none
if using onehot, define the columns to encode
"""
def prepareData(df,encoding='basic',encColumns=[]):
    df.drop("id", axis = 1,inplace=True)
    df = df[df['width'] != 10000000000]
    if encoding == 'basic':
        le = LabelEncoder()
        for column in nanColumns:
            df[column] = le.fit_transform(df[column])
    elif encoding == 'onehot':
        for column in encColumns: 
            one_hot = pd.get_dummies(df[column])
            dfEnc=pd.concat([df,one_hot],axis=1)
            dfEnc.drop(column,axis=1,inplace=True)
            df = dfEnc
    else:
        pass
    return df

def normalizeData(df,n_axis=0):
    return pd.DataFrame(normalize(df,axis=n_axis),columns=df.columns)    
    

def removeOutliers(df):
    #TODO
    pass

# useless, it takes one line anyway
def splitData():
    return ()

if __name__ == '__main__':
    df = loadData()
    df = prepareData(df)
    df = normalizeData(df)
    