#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 02:22:49 2020

@author: gabrielblins
"""


import pandas as pd

base = pd.read_csv('census.csv')

previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_previsores = LabelEncoder()

#labels = labelencoder_previsores.fit_transform(previsores[:,1])
#conversao de variaveis categoricas dos previsores para valores numericos
previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labelencoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labelencoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labelencoder_previsores.fit_transform(previsores[:,13])

#conversao das variaveis categoricas ja numericas para variaveis dummy (o valor associado não afetara por ser maior, dividindo em arrays de checagem do tipo)
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

#conversao da variavel da classe para um valor numerico (=< 50 -> 0, >50 -> 1)
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

#Escalonamento dos previsores (incluindo variaveis dummy)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

previsores = scaler.fit_transform(previsores)
















