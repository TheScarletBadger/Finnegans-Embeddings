# -*- coding: utf-8 -*-
"""
Trains and saves finnegan model
"""
#from gensim.models import Word2Vec

import re
import enchant
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import pandas as pd
from numpy import sqrt

#load english dictionary
d = enchant.Dict("en_GB")

#load trained model from disk
model = KeyedVectors.load('finnegan_keyed_vectors.kv')

#load list of finneganisms
with open('finneganisms.txt') as file:
    finneganisms = [line.rstrip() for line in file]

#fit scaler and perform pca over scaled embeddings
scaler = StandardScaler()
scaler.fit(model.vectors)
pca = PCA(n_components=2)
pca.fit(scaler.transform(model.vectors))

targetword = 'riverrun'
targloc = pca.transform(scaler.transform(model[targetword].reshape(1, -1)))
targdf = pd.DataFrame(model.most_similar(targetword,topn=10000),columns=['label','sim'])
targdf['english'] = targdf['label'].apply(d.check)
targdf['finneganism'] = targdf['label'].isin(finneganisms)
targdf[["pc1","pc2"]] = pca.transform(scaler.transform(model[targdf['label']]))
targdf["targdist"] = sqrt((targdf['pc1'] - targloc[0][0])**2 + (targdf['pc2'] - targloc[0][1])**2)
targdf = targdf[targdf['english'] | targdf['finneganism']]



