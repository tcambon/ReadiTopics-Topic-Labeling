#!/usr/bin/env python
# coding: utf-8

# # Topic Modeling with Gensim

# ### Installation du package pour gérer les stop words

# In[1]:


import nltk; nltk.download('stopwords')


# ### Importation des packages pour gérer les données et les visualiser

# In[2]:


import re
import numpy as np
import pandas as pd
from pprint import pprint
import collections
import math
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#scipy
import scipy 

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import os


# ### Preparation des stopwords

# In[3]:


# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# ### Importation des données

# In[4]:


# Import Dataset
os.chdir("C:/Users/thoma/Documents/Stage Cambon/Donnees/")


# In[5]:


# Import Dataset
df = pd.read_csv("data.txt",header=None, names=["content"])

df


# ### Conversion en liste

# In[6]:


# Convert to list
data = df.content.tolist()

###creation liste entiere du corpus ###
#l=[]

#for i in range(0,len(df)):
#    l.append(data[i])
#print(l)

#s = " "
#print(s.join(l))
#sdf=s.join(l)

#pprint(data[0])


# ### Séparation de chaque mots (token)

# In[7]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

#data_wordEntier= list(sent_to_words(sdf))


#print(data_words[:1])


# In[ ]:





# ### Création de Bigram et Trigram modèles

# In[8]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])


# ### Suppression des Stopwords et création de bigram et de lemmatization

# In[9]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[10]:


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#print(data_lemmatized[:1])


print(data_lemmatized[0:3])


# In[11]:


l=[]

for i in range(0,len(data_lemmatized)):
    l=l+data_lemmatized[i]


# In[12]:


l[:10]


# ### Création de dictionnaire et de corpus

# In[13]:


# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

corpus[:2]


# In[14]:


#Convert a streamed corpus in bag-of-words format into a sparse matrix scipy.sparse.csc_matrix, with documents as columns.

term_doc = gensim.matutils.corpus2csc(corpus)
freq = term_doc.sum(axis=1)
freq = (freq/np.sum(freq)).T
print(freq)


# In[15]:


print(term_doc)


# In[16]:


print(type(corpus[0]))


# ### Frequences de chaque mots

# In[17]:



len(texts)


# In[18]:


id2word[0]


# In[19]:


#[[freq for id, freq in cp] for cp in corpus]


# ### 13) Construction du modèle des topics

# In[20]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[21]:


pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# ### Score de cohérence

# In[22]:


# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# ### Visualisation des topics

# In[23]:


# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis


# ### 17) Recherche du nombre de topics optimal

# In[24]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print(num_topics)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[25]:


# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=3, limit=20, step=1)


# In[26]:


pprint(coherence_values)


# In[27]:


# Show graph
limit=20; start=3; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[28]:


# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

Num Topics = 2  has Coherence Value of 0.4451
Num Topics = 8  has Coherence Value of 0.5943
Num Topics = 14  has Coherence Value of 0.6208
Num Topics = 20  has Coherence Value of 0.6438
Num Topics = 26  has Coherence Value of 0.643
Num Topics = 32  has Coherence Value of 0.6478
Num Topics = 38  has Coherence Value of 0.6525
# In[29]:


# Select the model and print the topics
optimal_model = model_list[5]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


# ### Recherche du topic dominant par phrase

# In[30]:


optimal_model.get_topic_terms(1,10)

get_topics()
Get the term-topic matrix learned during inference.

Returns:	The probability for each word in each topic, shape (num_topics, vocabulary_size).
Return type:	numpy.ndarray
# In[31]:


df=pd.DataFrame(data=optimal_model.get_topics())


# In[32]:


df


# ### Importation des Candidats

# In[33]:


# Import Dataset
os.chdir("C:/Users/thoma/Documents/Stage Cambon/Donnees/")
dfCandidat = pd.read_csv('nounPhraseDBLP.txt', header=None,names=["content"])
lCandidat = dfCandidat.content.tolist()
lCandidat=np.asarray(lCandidat)
lCandidat[:10]


# In[34]:


#type(lCandidat)


# In[35]:


df


# ### On crée la fréquence pour chaque mot sur l'ensemble du corpus

# In[36]:


# theta matrice des freq sachant le topic
theta=df.values

#theta=np.transpose(theta)
theta.shape


# In[37]:


freq.shape


# In[38]:


print(freq)


# In[39]:


# on le repete 8 fois pour avoir la meme taille que la matrice theta
freq=np.ones((8,1))*freq


# In[40]:


freq


# In[41]:


print(theta)


# ### Fonction qui retourne les 10 meilleurs score

# In[42]:


def top10Score(score,lcandidat):
    """
    Return arrays that contain info about the top 10 score, in descending order

    Parameters:
    ----------
    score : Array containing the score of each label candidate
    lcandidat: Array containing each label candidate
    Returns:
    -------
    topValue : Array containing score of top 10 topic models in descending order
    topCandidate : Name of the top 10 score candidate in descending order
    """
    
    #on récupère l'indexe des 10 meilleurs score (dans les desordre)
    i=np.argpartition(score,len(score)-10)[-10:]

    # on recupere les indices qui trieraient les nombres dans le bonne ordre (ils vont de 0 à 9)
    ind=score[i].argsort()

    #on inverse la liste pour faire du plus grand au plus petit
    indf=np.flip(ind)
    
    topValue=(score[i])[indf]
    topCandidate=(lCandidat)[i][indf]
    
    return topValue, topCandidate


# ### Fonction qui retourne les tuples (Candidat,score) des 10 meilleurs score

# In[182]:


def top10ScoreCandidat(score,lcandidat):
    """
    Return arrays that contain info about the top 10 score, in descending order

    Parameters:
    ----------
    score : Array containing the score of each label candidate
    lcandidat: Array containing each label candidate
    Returns:
    -------
    topValue : Array containing score of top 10 topic models in descending order
    topCandidate : Name of the top 10 score candidate in descending order
    """
    
    #on récupère l'indexe des 10 meilleurs score (dans les desordre)
    i=np.argpartition(score,len(score)-10)[-10:]

    # on recupere les indices qui trieraient les nombres dans le bonne ordre (ils vont de 0 à 9)
    ind=score[i].argsort()

    #on inverse la liste pour faire du plus grand au plus petit
    indf=np.flip(ind)
    
    topValue=(score[i])[indf]
    topCandidate=(lCandidat)[i][indf]
    dicti=list(zip(topCandidate, topValue))
    
    return dicti


# In[180]:


l1=[1,2,3]
l2=['a','b','c']
l3=list(zip(l1,l2))

l3


# ### Zero-Order

# In[43]:


k=3


# In[183]:


def zero_order(freq,theta,lcandidat,NumTopic):
    """
    Calculate the Zero-Order Relevance

    Parameters:
    ----------
    freq : Array containing the frequency of occurrences of each word in the whole corpus
    theta : Array containing the frequency of occurrences of each word in each topic
    lcandidat: Array containing each label candidate
    NumTopic : The number of the topic
    
    Returns:
    -------
    topCandidate : Array containing the name of the top 10 score candidate for a given topic
    """
    
    #W matrice qui contient le score de chaque mot pour chaque topic
    W=np.log(theta/freq)
    
    # score des tous les candidats pour le topic NumTopic
    score=np.array([])
    
    for indice in range (len(lCandidat)):
        candidat=lCandidat[indice].split(" ")
        i=id2word.doc2idx(candidat)
        # supprime les -1 (qui signifie pas trouvé)
        i[:] = [v for v in i if v != -1]
        
        score=np.append(score,np.sum(W[k,i]))
        
    #topValue, topCandidate = top10Score(score,lCandidat)
    dicti=top10ScoreCandidat(score,lcandidat)
  
    return dicti


# In[45]:


#topValue, topCandidate = top10Score(score,lCandidat)


# In[168]:


#topValue, topCandidate = zero_order(freq,theta,lCandidat,k)


# In[169]:


#topValue


# In[170]:


#topCandidate


# In[184]:


zero_order(freq,theta,lCandidat,k)


# ### M order

# In[172]:


def m_order(freq,theta,lcandidat,k):
    """
    Calculate the M-Order Relevance

    Parameters:
    ----------
    freq : Array containing the frequency of occurrences of each word in the whole corpus
    theta : Array containing the frequency of occurrences of each word in each topic
    lcandidat: Array containing each label candidate
    k : The number of the topic
    
    Returns:
    -------
    topCandidate : Array containing the name of the top 10 score candidate for a given topic
    """
    # bob la liste qui contient tous les numéros de topic (lignes), sauf celui du topic selectionné
    bob =list(range(8))
    del bob[k]
    
    # M la moyenne des proba d'avoir le mot, sachant qu'on est pas dans le topic k
    M=theta[bob,:].mean(axis=0)
    
    #W matrice qui contient le score de chaque mot pour chaque topic
    W=np.log(theta[k]/M)
    
    score=np.array([])
    for indice in range (len(lCandidat)):
        # On sépare chaque terme candidat en mots
        candidat=lCandidat[indice].split(" ")
        # i est une liste qui contient les numéros des mots qui sont candidats
        i=id2word.doc2idx(candidat)
        # supprime les -1 (qui signifie pas trouvé)
        i[:] = [v for v in i if v != -1]
        
        # Score contient la somme des scores des mots du meme candidat
        score=np.append(score,np.sum(W[i]))
    #On appel la fonction qui trie dans le bonne ordre
    #topValue, topCandidate = top10Score(score,lCandidat)
    
    dicti=top10ScoreCandidat(score,lcandidat)

    return dicti


# In[173]:


#topValue, topCandidate = m_order(freq,theta,lCandidat,k)


# In[51]:


#topValue, topCandidate = top10Score(score,lCandidat)


# In[174]:


#topValue


# In[175]:


#topCandidate


# In[185]:


m_order(freq,theta,lCandidat,k)




# # First-order
## ne fonctionne pas encore
# In[56]:


n=len(lCandidat[0].split(" "))
wsl=1/n
wsl


# In[57]:


wsl=np.ones(n)*wsl


# In[61]:


l=id2word.doc2idx(lCandidat[0].split(" "))
# supprime les -1 (qui signifi pas trouvé)
print(l)
while -1 in l:
    l.remove(-1)
    wsl= wsl[:-1]
wsl


# In[ ]:


# k le topic 
k=0

# score des tous les candidats pour le topic NumTopic
score=np.array([])

for indice in range (len(lCandidat)):
    candidat=lCandidat[indice].split(" ")
    #print("candidat: ",candidat)
    n=len(lCandidat[indice].split(" "))
    wsl=1/n
    wsl=np.ones(n)*wsl
    
    i=id2word.doc2idx(candidat)
    # supprime les -1 (qui signifi pas trouvé)
    while -1 in i:
        i.remove(-1)
        #wsl c'est la proba théorique de chaque mot dans le label, qui est la meme
        wsl=wsl[:-1]
    wst= theta[k,i]    
    score=np.append(score,-(scipy.stats.entropy(wst,wsl)))
    print("indice ",indice)


# In[66]:


import time
import random


# In[157]:


l=[-1,2,-1,8]*1000


# In[158]:


then = time.time() #Time before the operations start


# In[159]:


l[:] = [v for v in l if v != -1]


# In[160]:


now = time.time() #Time after it finished

print("It took: ", now-then, " seconds")


# In[161]:


l=[-1,2,-1,8]*1000


# In[162]:


then = time.time() #Time before the operations start


# In[163]:


while -1 in l:
    l.remove(-1)


# In[164]:


now = time.time() #Time after it finished

print("It took: ", now-then, " seconds")

