# -*- coding: utf-8 -*-
"""
Antonis Michalas. (2019).
Text files from Gutenberg database [Data set].
Zenodo. https://doi.org/10.5281/zenodo.3360392

https://archive.org/stream/finneganswake00joycuoft/finneganswake00joycuoft_djvu.txt
"""
#from gensim.models import Word2Vec
import nltk
import re
from os import listdir
import gc
from gensim.models import Word2Vec
from multiprocessing import cpu_count

cores = cpu_count()
nltk.download('punkt_tab')
cnt = 1
datadir = 'D184MB+FW/'
corpus = ''
for filename in listdir(datadir):
    print(f'Processing file {cnt}: {filename}')
    f = open(datadir + filename ,'r')
    try:
        corpus = corpus + f.read()
    except:
        print(f'Error reading file {filename} Skipping')
    f.close()
    cnt=cnt+1

print('Cleaning tokens...')    
#remove newline characters
corpus = corpus.replace('\n', ' ').lower()

#remove characters that are not letters, spaces or periods
cleaned = re.sub(r"[^A-Za-z\s.'-]", "", corpus)
del(corpus)
gc.collect()

#split corpus into list sentences
sentences = nltk.sent_tokenize(cleaned)
del(cleaned)
gc.collect()

#split each item in sentence list into list of words
words = [nltk.tokenize.word_tokenize(i) for i in sentences]
del(sentences)
gc.collect()

print('Training model...')    
#initialize and train model
model = Word2Vec(vector_size=500, window=10, workers=cores, min_count=1, seed=42)
model.build_vocab(words)
model.train(words, total_examples=len(words), epochs=25)
model.save('finnegan.model')

#print closest words to man and woman in embedding space
print(model.wv.most_similar('man'))
print(model.wv.most_similar('woman'))

#word arithmetic test, should return 'she' as most probable
print(model.wv.most_similar(positive=['he', 'woman'],negative=['male'],topn=1))
#word arithmetic test, should return 'he' as most probable
print(model.wv.most_similar(positive=['she', 'man'],negative=['female'],topn=1))