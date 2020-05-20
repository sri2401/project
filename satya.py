# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:13:14 2020

@author: hp
"""


import pickle
loaded_model = pickle.load(open('model.sav', 'rb'))
result = loaded_model.predict([[7,	289.36,	0.0,	0.0,	75,	1,	2012,	2,	10,	10,	0]])
print(result)