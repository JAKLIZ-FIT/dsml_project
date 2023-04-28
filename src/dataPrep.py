# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:19:30 2023

@author: jiri zilka
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def showCorrMatrix():
    pass 

def prepareData(df):
    

if __name__ == '__main__':
    df = pd.read_csv('../data/manufacturing.csv')
    
    df = prepareData(df)
    
    showCorrMatrix(df)
    