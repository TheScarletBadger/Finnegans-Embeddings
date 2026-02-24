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
from numpy import sqrt, where

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
    targdf['Word Type'] = where(targdf['Finneganism'], "Finneganism", "English")
    targdf2 = targdf[['Word','Similarity','Word Type']]
    return targdf2


with gr.Blocks() as app:
    gr.HTML("<h1>Finnegan's Embeddings</h1>")
    gr.HTML("This app maps the semantic similarity between both English words and 'Finneganisms' (neologisms from James Joyce’s famously impenetrable novel Finnegan’s Wake). Searching for a word returns a dataframe of words ordered by semantic similarity to the target word. Relative locations of words in embedding space are approximated and visualized by using a principal component analysis to reduce the 250 dimensional embedding vectors to 2 dimensions and projecting word vectors onto these principal component axes.")
    with gr.Row():
        targetbox = gr.Textbox(label='Target Word (type word and press enter key to perform search)')
    with gr.Accordion("Table of words most semantically similar to target"):
        df1 = gr.Dataframe(row_count = 6)
    with gr.Accordion("Locations of semantically similar words in embedding space approximated using first 2 principal components."):
        plot1 = gr.Plot()
    targetbox.submit(fn=findnearest, inputs=targetbox, outputs=df1)

app.launch(server_name="0.0.0.0", server_port= 7860)
