# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:19:30 2023

@author: jiri zilka
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, normalize

# columns that need to be encoded
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
            df.loc[:,column] = le.fit_transform(df.loc[:,column])
    elif encoding == 'onehot':
        for column in encColumns: 
            one_hot = pd.get_dummies(df[column])
            dfEnc=pd.concat([df,one_hot],axis=1)
            dfEnc.drop(column,axis=1,inplace=True)
            df = dfEnc
    else:
        pass
    return df

def normalizeData(df,n_axis=0, type='norm'):
    if type == 'norm':
        return pd.DataFrame(normalize(df,axis=n_axis),columns=df.columns)    
    elif type == 'minMax':
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        df = pd.DataFrame(min_max_scaler.fit_transform(df),columns=df.columns)
    else:
        return df

def removeOutliers(df, useMedian=False):
    if useMedian:
        df.loc[df['width'] == 10000000000,'width'] = df.width.median()
    else: 
        df = df[df['width'] != 10000000000]
    return df

# split data into train set and test set 
def splitData(X,Y):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, Y, test_size=0.25,random_state=42)

if __name__ == '__main__':
    df = loadData()
    df = prepareData(df)
    df = normalizeData(df)
    