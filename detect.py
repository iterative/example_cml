#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:21:05 2022

@author: primous.pomalegni
"""

import numpy as np
import pickle

X_test = np.genfromtxt("data/test_features.csv")

# load
with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)
    
r = clf.predict(X_test[0:15])

with open("detect.txt", "w") as outfile:
    outfile.write("Detection: " + str(r) + "\n")

print(r)