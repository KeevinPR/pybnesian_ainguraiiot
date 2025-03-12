#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
os.environ["OMP_NUM_THREADS"] = '10'
import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("asia10K.csv", sep=None, engine='python', na_values='?')
df = pd.read_csv("abalone.data", sep=None, engine='python', na_values='?')
df = pd.read_csv("/home/juan/Documents/UPM/Aingura/P0065_Huella_Bombas_32768_3padres/bombas_aena_sample_train_32k_3pa.csv", sep=None, engine='python', na_values='?')
index_constant = np.where(df.nunique() == 1)[0]
constant_columns = [df.columns[i] for i in index_constant]
df = df.drop(columns=constant_columns)
df = df.dropna()
cat_data = df.select_dtypes('object').astype('category')
for c in cat_data:
    df = df.assign(**{c: cat_data[c]})
float_data = df.select_dtypes('number').astype('float64')
for c in float_data:
    df = df.assign(**{c: float_data[c]})




import pybnesian as pbn




mskcmi = pbn.MixedKMutualInformation(df=df, k=20, samples=10, scaling="min_max",gamma_approx=True, adaptive_k=True)
rcot = pbn.RCoT(df=float_data)
# rcot = pbn.LinearCorrelation(df=float_data)
# rcot = pbn.MutualInformation(df=df)
pdag = pbn.PC().estimate(hypot_test=rcot, use_sepsets=False, arc_blacklist = [], arc_whitelist = [], edge_blacklist = [], edge_whitelist = [], verbose = 1)
print(pdag.edges(), pdag.arcs())
