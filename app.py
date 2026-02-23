# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 22:42:22 2026
"""

import gradio as gr
import matplotlib.pyplot as plt
import enchant
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import pandas as pd
from numpy import sqrt

gr.close_all()

#load english dictionary
d = enchant.Dict("en_GB")

#load trained model from disk
model = Word2Vec.load('finnegan.model')

#load list of finneganisms
with open('finneganisms.txt') as file:
    finneganisms = [line.rstrip() for line in file]

#fit scaler and perform pca over scaled embeddings
scaler = StandardScaler()
scaler.fit(model.wv.vectors)
pca = PCA(n_components=2)
pca.fit(scaler.transform(model.wv.vectors))

def findnearest(targetword):
    targloc = pca.transform(scaler.transform(model.wv[targetword].reshape(1, -1)))
    targdf = pd.DataFrame(model.wv.most_similar(targetword,topn=1000),columns=['label','sim'])
    targdf['english'] = targdf['label'].apply(d.check)
    targdf[["pc1","pc2"]] = pca.transform(scaler.transform(model.wv[targdf['label']]))
    targdf["targdist"] = sqrt((targdf['pc1'] - targloc[0][0])**2 + (targdf['pc2'] - targloc[0][1])**2)
    return targdf


with gr.Blocks() as app:
    with gr.Row():
        with gr.Column(scale=1):
            targetbox = gr.Textbox(label='Target Word')
            button = gr.Button(value="Start")
        with gr.Column(scale=4):
            plot1 = gr.Plot()
    button.click(fn=findnearest, inputs=targetbox.value)
    

app.launch(server_name="0.0.0.0", server_port= 7860)
