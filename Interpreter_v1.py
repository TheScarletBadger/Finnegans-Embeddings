# -*- coding: utf-8 -*-
"""
Trains and saves finnegan model
"""
#from gensim.models import Word2Vec

import re
import enchant
d = enchant.Dict("en_GB")


from gensim.models import Word2Vec

d = enchant.Dict("en_GB")


model = Word2Vec.load('finnegan.model')

targetword = 'bisectualism'

x = [[a,b] for a,b in model.wv.most_similar(targetword,topn=1000) if d.check(a)]


