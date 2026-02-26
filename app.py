# -*- coding: utf-8 -*-
"""
Gradio App for interacting with keyed vector files.
https://huggingface.co/spaces/UltimaBadger/Finnegans-Embeddings
"""

import gradio as gr
import plotly.express as px
import nltk
from nltk.corpus import words
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import pandas as pd
from numpy import sqrt, where
import numpy as np
from huggingface_hub import hf_hub_download

nltk.download('words')

gr.close_all()

model_path = hf_hub_download(repo_id="UltimaBadger/Finnegan-Word2Vec", filename="finnegan_keyed_vectors.kv")
vectors_path = hf_hub_download(repo_id="UltimaBadger/Finnegan-Word2Vec", filename="finnegan_keyed_vectors.kv.vectors.npy")

#load trained model
model = KeyedVectors.load(model_path)
model.vectors = np.load(vectors_path) 

#load list of finneganisms
with open('finneganisms.txt') as file:
    finneganisms = [line.rstrip() for line in file]

#fit scaler and perform pca over scaled embeddings
scaler = StandardScaler()
scaler.fit(model.vectors)
pca = PCA(n_components=2)
pca.fit(scaler.transform(model.vectors))

def plot_embeddings(pca_df,targetword):
    # Create the plot
    targloc = pca.transform(scaler.transform(model[targetword].reshape(1, -1)))
    targlocdf = pd.DataFrame(data={"PC1": [targloc[0][0]], "PC2": [targloc[0][1]], "Word":[targetword],"Word Type":["Target"]})
    pca_df = pd.concat([pca_df,targlocdf],join='inner', ignore_index=True)
    fig = px.scatter(pca_df, x="PC1", y="PC2", text="Word", color="Word Type", template="plotly_dark")
    fig.update_traces(textposition='top center')
    fig.update_traces(marker=dict(size=50,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))
    fig.show()
    return fig

def findnearest(targetword,filters,nr):
    targetword = targetword.lower()
    targloc = pca.transform(scaler.transform(model[targetword].reshape(1, -1)))
    targdf = pd.DataFrame(model.most_similar(targetword,topn=10000),columns=['Word','Similarity'])
    targdf['English'] = targdf['Word'].isin(words.words())
    targdf['Finneganism'] = targdf['Word'].isin(finneganisms)
    targdf[["PC1","PC2"]] = pca.transform(scaler.transform(model[targdf['Word']]))
    targdf["Distance (PC space)"] = sqrt((targdf['PC1'] - targloc[0][0])**2 + (targdf['PC2'] - targloc[0][1])**2)
    targdf = targdf[targdf['English'] | targdf['Finneganism']]
    targdf['Word Type'] = where(targdf['Finneganism'], "Finneganism", "English")
    targdf = targdf[targdf['Word Type'].isin(filters)]
    targdf = targdf.head(nr)
    targdf2 = targdf[['Word','Similarity','Word Type']]
    fig = plot_embeddings(targdf,targetword)
    return targdf2, fig

with gr.Blocks() as app:
    gr.HTML('''
        <h1>Finnegan's Embeddings</h1>
        <h4>"riverrun, past Eve and Adam's, from swerve of shore to bend of bay, brings us by a commodius vicus of recirculation back to Howth Castle and Environs."</h4>
        <p>This model seeks to answer the question "What if we trained a Word2Vec model to map semantic relationships between words then drove it insane by adding James Joyce's Finnegan's Wake to its training corpus?"</p>
        ''')
    with gr.Accordion("About this app"):
        gr.HTML('''
            <p>The original objective in training this model was to see if it was possible to find semantic relationships between Joyce's neologisms and common english words. However the low frequency of the neologisms with only one or two examples of each in Joyce's work effectively precludes building meanigful embeddings for them. Nevertheless it remains a fun experiment. A Word2Vec model was trained on a 670MB subset of the <a href="https://zenodo.org/records/3360392">Gutenberg text corpus</a> augmented with a copy of <a href="https://archive.org/stream/finneganswake00joycuoft/finneganswake00joycuoft_djvu.txt">Finnegan's Wake</a> using the <a href="https://pypi.org/project/gensim/">Gensim library</a>. Words are represented as 250-dimension embedding vectors. Semantic relationships between words can be determined by examining the proximity of words in embedding space. </p>
            <p>Searching for a target word returns a dataframe of words ordered by semantic similarity to the target word. Relative locations of words in embedding space are approximated and visualized by using a principal component analysis to reduce the 250-dimensional embedding vectors to 2 dimensions and projecting word vectors onto these principal component axes. This is an incredibly lossy transformation, so some discrepancy between the similarity scores and proximity in principal component space is expected. </p>
                ''')
    with gr.Row():
        targetbox = gr.Textbox(label='Target Word (type word and press enter key to perform semantic similarity search)',value='test',interactive=True)
        filt = gr.CheckboxGroup(choices=['English', 'Finneganism'],value=['English', 'Finneganism'],label='Filter word type',interactive=True)
        nresults = gr.Slider(minimum=1,maximum=100,value=5,interactive=True,precision=0,step=1,label='Number of results to display')
    with gr.Accordion("Table of words most semantically similar to target"):
        df1 = gr.Dataframe(row_count = 5)
    with gr.Accordion("Locations of semantically similar words in embedding space approximated using first 2 principal components."):
        plot1 = gr.Plot()
    targetbox.submit(fn=findnearest, inputs=[targetbox,filt,nresults], outputs=[df1,plot1])
    filt.change(fn=findnearest, inputs=[targetbox,filt,nresults], outputs=[df1,plot1])
    nresults.change(fn=findnearest, inputs=[targetbox,filt,nresults], outputs=[df1,plot1])
    gr.on(triggers=[app.load],fn=findnearest, inputs=[targetbox,filt,nresults], outputs=[df1,plot1])

app.launch()