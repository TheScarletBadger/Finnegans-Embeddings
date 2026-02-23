# -*- coding: utf-8 -*-
"""
Trains and saves finnegan model
"""
#from gensim.models import Word2Vec
import nltk
import re
from os import listdir
from gensim.models import Word2Vec
from multiprocessing import cpu_count
import enchant

#load english dictionary
d = enchant.Dict("en_GB")

#import text data into mega-string, very inefficient but this is a one off...
cores = cpu_count()
nltk.download("punkt_tab")
cnt = 1
datadir = "gutenberg/"
gutenberg = ""
for filename in listdir(datadir):
    print(f"Processing Gutenberg corpus - file {cnt}: {filename}")
    f = open(datadir + filename ,"r")
    try:
        gutenberg = gutenberg + f.read()
    except:
        print(f"Error reading file {filename} Skipping")
    f.close()
    cnt=cnt+1

print("Cleaning and tokenizing...")  

#Clean up text
gutenberg = gutenberg.replace("\n", " ").lower()
gutenberg_cleaned = re.sub(r"[^A-Za-z\s.]", "", gutenberg)
gutenberg_cleaned = re.sub(r'\s+', ' ', gutenberg_cleaned)

#split gutenberg corpus into list sentences
gutenberg_sentences = nltk.sent_tokenize(gutenberg_cleaned)

#remove periods
gutenberg_sentences = [s.replace(".", "") for s in gutenberg_sentences if s.replace(".", "") != '']

#split each item in sentence list into list of words
gutenberg_words = [nltk.tokenize.word_tokenize(i) for i in gutenberg_sentences if len(i) > 1]

print('Removing sentences containing non-english words from the Gutenberg corpus')
#remove non-english words
gutenberg_words = [sen for sen in gutenberg_words if all(d.check(wrd) for wrd in sen)]


gutenberg_set = set([wrd for sen in gutenberg_words for wrd in sen])

print("Processing Finnegan's Wake")
f = open("finnegan.txt" ,"r")
finnegan = f.read()
f.close()

print("Cleaning and tokenizing...")

#Clean up text
finnegan_cleaned = finnegan.replace("- \n", "").lower()
finnegan_cleaned = finnegan_cleaned.replace("\n", " ")
finnegan_cleaned = re.sub(r'\s+', ' ', finnegan_cleaned)
finnegan_cleaned = re.sub(r"[^A-Za-z\s.]", "", finnegan_cleaned)

#split finnegans wake into list sentences
finnegan_sentences = nltk.sent_tokenize(finnegan_cleaned)

#split each item in sentence list into list of words
finnegan_words = [nltk.tokenize.word_tokenize(i) for i in finnegan_sentences if len(i) > 1]

#identify words unique to finnegans wake and write to file
finnegan_set = set([wrd for sen in finnegan_words for wrd in sen])
finneganisms = finnegan_set - gutenberg_set
file = open('finneganisms.txt','w')
for i in finneganisms:
	file.write(i+"\n")
file.close()

#merge gutenberg and finnegans wake tokens to form training corpus
corpus = finnegan_words + gutenberg_words

print('Training model...')    
#initialize and train model
model = Word2Vec(vector_size=500, window=10, workers=cores, min_count=1, seed=42)
model.build_vocab(corpus)
model.train(corpus, total_examples=len(corpus), epochs=25)
model.save('finnegan.model')

print('Running post-training tests')
#print closest words to man and woman in embedding space
print(model.wv.most_similar('man'))
print(model.wv.most_similar('woman'))

#word arithmetic test, should return 'she' as most probable
print(model.wv.most_similar(positive=['he', 'woman'],negative=['male'],topn=1))
#word arithmetic test, should return 'he' as most probable
print(model.wv.most_similar(positive=['she', 'man'],negative=['female'],topn=1))
#word arithmetic test, should return 'king' as most probable
print(model.wv.most_similar(positive=['queen', 'man'],negative=['woman'],topn=1))
#word arithmetic test, should return 'queen' as most probable
print(model.wv.most_similar(positive=['king', 'woman'],negative=['man'],topn=1))

testword = 'bababadalgharaghtakamminarronnkonnbronntqnnerronntuonnthunntrovarrhounawnskawntoohoohoordenenthurnuk'
x = set([word for word, _ in model.wv.most_similar(testword,topn=250)])
print('')
print(x-finneganisms)

