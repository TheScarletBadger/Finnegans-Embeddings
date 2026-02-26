# Finnegans-Embeddings
"riverrun, past Eve and Adam's, from swerve of shore to bend of bay, brings us by a commodius vicus of recirculation back to Howth Castle and Environs."

This model seeks to answer the question "What if we trained a Word2Vec model to map semantic relationships between words then drove it insane by adding James Joyce's Finnegan's Wake to its training corpus?"

The original objective in training this model was to see if it was possible to find semantic relationships between Joyce's neologisms and common english words. However the low frequency of the neologisms with only one or two examples of each in Joyce's work effectively precludes building meanigful embeddings for them. Nevertheless it remains a fun experiment. A Word2Vec model was trained on a 670MB subset of the Gutenberg text corpus augmented with a copy of Finnegan's Wake using the Gensim library. Words are represented as 250-dimension embedding vectors. Semantic relationships between words can be determined by examining the proximity of words in embedding space.
FE_Train_Model_v3.py trains a Word2Vec model and saves the keyed vector files. The script assumes that text files are placed in a directory called "Guternberg" which is co-located with the script file.

Pre-trained files created by this script are available below.
  https://huggingface.co/UltimaBadger/Finnegan-Word2Vec

App.py is a Gadio app used to query the keyed vectors. Searching for a target word returns a dataframe of words ordered by semantic similarity to the target word. Relative locations of words in embedding space are approximated and visualized by using a principal component analysis to reduce the 250-dimensional embedding vectors to 2 dimensions and projecting word vectors onto these principal component axes. This is an incredibly lossy transformation, so some discrepancy between the similarity scores and proximity in principal component space is expected.
  https://huggingface.co/spaces/UltimaBadger/Finnegans-Embeddings


