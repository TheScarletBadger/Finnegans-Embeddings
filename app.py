# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 22:42:22 2026
"""

import gradio as gr
import matplotlib.pyplot as plt
import enchant
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import pandas as pd
from numpy import sqrt

gr.close_all()

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

def findnearest(targetword):
    print(targetword)
    targetword = targetword.lower()
    targloc = pca.transform(scaler.transform(model[targetword].reshape(1, -1)))
    targdf = pd.DataFrame(model.most_similar(targetword,topn=10000),columns=['Word','Similarity'])
    targdf['English'] = targdf['Word'].apply(d.check)
    targdf['Finneganism'] = targdf['Word'].isin(finneganisms)
    targdf[["PC1","PC2"]] = pca.transform(scaler.transform(model[targdf['Word']]))
    targdf["Distance (PC space)"] = sqrt((targdf['PC1'] - targloc[0][0])**2 + (targdf['PC2'] - targloc[0][1])**2)
    targdf = targdf[targdf['English'] | targdf['Finneganism']]
    print(targdf)
    return targdf


with gr.Blocks() as app:
    with gr.Row():
        with gr.Column(scale=1):
            targetbox = gr.Textbox(label='Target Word')
        with gr.Column(scale=4):
            plot1 = gr.Plot()
            df1 = gr.Dataframe()
    targetbox.submit(fn=findnearest, inputs=targetbox, outputs=df1)

app.launch(server_name="0.0.0.0", server_port= 7860)
