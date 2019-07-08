# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:16:58 2019

@author: Thomas Cambon
"""

## entrée:
# freq de chaque mot pour l'ensemble du corpus theta
# liste des candidats lCandidat
# freq de chaque mot pour chaque topic


import numpy as np


def top10score(score,lcandidat):
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
    topCandidate=(lcandidat)[i][indf]
    
    return topValue, topCandidate



def zero_order(freq,theta,lcandidat,k, id2word):
    """
    Calculate the Zero-Order Relevance

    Parameters:
    ----------
    freq : Array containing the frequency of occurrences of each word in the whole corpus
    theta : Array containing the frequency of occurrences of each word in each topic
    lcandidat: Array containing each label candidate
    NumTopic : The value of the topic
    id2word : dictionnary between word and id ex : id2word = corpora.Dictionary(data_lemmatized)

    
    Returns:
    -------
    dicti : Array containing the name of the top 10 (name,score) candidate for a given topic
    """
    
    #W matrice qui contient le score de chaque mot pour chaque topic
    W=np.log(theta/freq)
    
    # score des tous les candidats pour le topic NumTopic
    score=np.array([])
    
    for indice in range (len(lcandidat)):
        candidat=lcandidat[indice].split(" ")
        i=id2word.doc2idx(candidat)
        # supprime les -1 (qui signifie pas trouvé)
        i[:] = [v for v in i if v != -1]
        
        score=np.append(score,np.sum(W[k,i]))
        
    #topValue, topCandidate = top10Score(score,lCandidat)
    dicti=top10score(score,lcandidat)
  
    return dicti

def m_order(freq,theta,lcandidat,k, id2word):
    """
    Calculate the M-Order Relevance

    Parameters:
    ----------
    freq : Array containing the frequency of occurrences of each word in the whole corpus
    theta : Array containing the frequency of occurrences of each word in each topic
    lcandidat: Array containing each label candidate
    k : The number of the topic
    id2word : dictionnary between word and id ex : id2word = corpora.Dictionary(data_lemmatized)

    Returns:
    -------
    dicti : Array containing the name of the top 10 (name,score) candidate for a given topic
    """
    # bob la liste qui contient tous les numéros de topic (lignes), sauf celui du topic selectionné
    bob =list(range(8))
    del bob[k]
    
    # M la moyenne des proba d'avoir le mot, sachant qu'on est pas dans le topic k
    M=theta[bob,:].mean(axis=0)
    
    #W matrice qui contient le score de chaque mot pour chaque topic
    W=np.log(theta[k]/M)
    
    score=np.array([])
    for indice in range (len(lcandidat)):
        # On sépare chaque terme candidat en mots
        candidat=lcandidat[indice].split(" ")
        # i est une liste qui contient les numéros des mots qui sont candidats
        i=id2word.doc2idx(candidat)
        # supprime les -1 (qui signifie pas trouvé)
        i[:] = [v for v in i if v != -1]
        
        # Score contient la somme des scores des mots du meme candidat
        score=np.append(score,np.sum(W[i]))
    #On appel la fonction qui trie dans le bonne ordre
    #topValue, topCandidate = top10Score(score,lCandidat)
    
    dicti=top10score(score,lcandidat)

    return dicti