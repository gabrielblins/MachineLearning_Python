#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 03:30:15 2020

@author: gabrielblins
"""


import pandas as pd

base = pd.read_csv('credit_data.csv')

base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()

imputer = imputer.fit(previsores[:,0:3])

previsores[:,0:3] = imputer.transform(previsores[:,0:3])

#Escalonamento de valores
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe,test_size = 0.25, random_state = 0)