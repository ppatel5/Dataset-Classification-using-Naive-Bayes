# -*- coding: utf-8 -*-

import os
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from math import log


dict={}
mytokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = PorterStemmer()
sortedstopwords = sorted(stopwords.words('english'))

#function that cleans data
def tokenize(doc):
    tokens = mytokenizer.tokenize(doc)
    lowertokens = [token.lower() for token in tokens]
    filteredtokens = [stemmer.stem(token) for token in lowertokens if not token in sortedstopwords]
    return filteredtokens

path = './20_newsgroups/'
#preparing train data
for dir in os.listdir(path):
    train=""
    corpus = path+dir       
    for filename in os.listdir(corpus)[:501]:
        train = train + " "
        with open(os.path.join(corpus,filename)) as file_object:
           train +=file_object.read()
        file_object.close()

    tokens = tokenize(train)
    tokens = Counter(tokens)
    wordcount=0
    wordcount +=sum([tokens.get(token) for token in tokens])
    dict[dir] = tokens
    dict[dir]['word_count'] = wordcount
#working with test data
positive=0
prediction={}
total_unique_test_words = 0
k=0
for dir in dict:
    total_unique_test_words+=((len(dict[dir].keys()))-1)
for dir in os.listdir(path):
    
    corpus = path+dir 
   
    for filename in os.listdir(corpus)[501:]:
        k+=1
        test = ""
        with open(os.path.join(corpus,filename)) as file_object:
           train =file_object.read()
        file_object.close()
        tokens_test = tokenize(train)
        
        prob_all={}
        
        for x in dict:
            prob=0.0
            for token in tokens_test:
                prob = (prob) + log((dict[x][token]+1)/(dict[x]['word_count']*total_unique_test_words))                
            prob_all[x] = prob/20

        with open(os.path.join(corpus,filename)) as file_object:
            prediction[filename] = max(prob_all, key=prob_all.get)
            if(prediction[filename]==dir):
                positive += 1
         
                
        file_object.close()
#print(prediction)
#print(positive)
Accuracy=(positive/10000) * 100
print(Accuracy)
#print(k)

