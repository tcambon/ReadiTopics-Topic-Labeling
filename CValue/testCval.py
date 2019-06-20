# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:45:29 2019

@author: Thomas Cambon
"""

#source: https://github.com/huanyannizu/C-Value-Term-Extraction

import pandas as pd

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import os
from NoName import NoName

noun = ['NN','NNS','NNP','NNPS']#tags of noun
adj = ['JJ']#tags of adjective
pre = ['IN']#tags of preposition


# source : https://github.com/huanyannizu/C-Value-Term-Extraction
os.chdir("C:/Users/thoma/Documents/Stage Cambon/Donnees/wikiTagged")
Data= 'MiniNounPhraseWiki02tag.txt' #input data file
#Data= 'nounPhraseWiki02tag.txt' #input data file


#f = open(Data,encoding='utf8').readlines()
#f
df = pd.read_csv(Data,header=None, names=["content"])
# Convert to list
f = df.content.tolist()
f


#lingui_filter= 'AdjNoun'
lingui_filter='AdjPrepNoun'
L=10 #the expected maximum length of a term
freq_threshold=0.2
CValue_threshold=0.2


#os.chdir('C:\\Users\\thoma')

candidate = candidate = dict([(key, []) for key in range(2,L+1)])
nblignes=len(f)
nb=0
for sentence in f:
    
    print("Ligne ",nb, " sur ",nblignes-1)
    sentence = sentence.rstrip('\n').split(' ') 
    n_words = len(sentence)
    start=0
    while start < n_words - 2:
        #print("start= ",start)
        #print("limite= ",n_words-2)
        i =  start
        noun_ind = []
        pre_ind = []
        pre_exist = False
        while True:
            #print("limite = i<",n_words-2)
            word = NoName()
            #print("i= ", i)
            if (i>=n_words-2):
                break
            word.word(sentence[i])
            #print("word : ",word.word)
            #print("tag : ",word.tag)

            if word.tag in noun:
                noun_ind.append(i)
                i += 1
            elif (lingui_filter == ('AdjNoun' or 'AdjPrepNoun')) and word.tag in adj:
                word_next = NoName()
                word_next.word(sentence[i+1])
                if word_next.tag in noun: 
                    noun_ind.append(i+1)
                    i += 2
                elif word_next.tag in adj:
                    i += 2
                else: 
                    end = i+1
                    break
            elif (lingui_filter == 'AdjPrepNoun') and not pre_exist and i != 0 and (word.tag in pre):
                pre_ind.append(i)
                pre_exist = True
                i += 1
            else: 
                end = i
                break

        if len(noun_ind) != 0:
           #print("noun_ind = ",noun_ind)
            #print("pre_ind = ", pre_ind)
            for i in list(set(range(start,noun_ind[-1]))-set(pre_ind)):
                for j in noun_ind:
                    if j-i >= 1 and j-i <= L-1:
                        substring = NoName()
                        substring.substring(sentence[i:j+1])
                        #print("substring : ",substring.substring(sentence[i:j+1]))
                        exist = False
                                                
                        for x in candidate[j-i+1]:
                            #print("x= ",x)
                            #print("x.words: ",x.words)
                            #print("substring.words: ",substring.words)
                            if x.words == substring.words:
                                x.f += 1
                                exist = True
                        if not exist:
                            #print("not exist")
                            #print("index: ",j-i+1)
                            #print("substring : ",substring.words)
                            #candidate[j-i+1].append(" ".join(substring.words))
                            candidate[j-i+1].append(substring)
                            substring.f = 1
        start =  start + 1
    nb+=1
        
##Compute C-Value##################################################################################
print("Calcul de la C-value")        
Term = []           
for l in reversed(range(2,L+1)):
    candi = candidate[l]
    for phrase in candi:
        if phrase.c == 0:
            phrase.CValue_non_nested()
        else: phrase.CValue_nested()         
        
        if phrase.CValue >= CValue_threshold: 
            Term.append(phrase)
            for j in range(2,phrase.L):
                for i in range(phrase.L-j+1):
                    substring = phrase.words[i:i+j]
                    for x in candidate[j]:
                        if substring == x.words:
                            x.substringInitial(phrase.f)
                            for m in Term:
                                if ' '.join(x.words) in ' '.join(m.words): 
                                    x.revise(m.f,m.t)
                                                            
Term.sort(key=lambda x: x.CValue, reverse=True)
print("cval: ",len(Term))
print("cval: ",Term[-1].CValue)

print("cval: ",Term[0].CValue)
print("cval: ",Term[0].CValue)

print("cval: ",Term[0].words)

score=[]
for i in range(0,len(Term)-1):
    score.append([" ".join(Term[i].words),Term[i].CValue])
    