#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 01:18:24 2020

@author: gabrielblins
"""


import pandas as pd

base = pd.read_csv('credit_data.csv')
base.describe()

base.loc[base['age'] < 0]
#apagar coluna
base.drop('age', 1, inplace=True)
#apagar registros problematicos
base.drop(base[base.age < 0].index, inplace=True)
#alterar manualmente
#preencher com a media
base.mean()
base['age'].mean()
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()

#Registros Nulos
#Formas de encontrar
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

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




















