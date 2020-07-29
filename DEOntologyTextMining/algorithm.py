#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 03:49:00 2020
Load trained skipg model and apply defined algorithms to extract ontological components from text
@author: weinie
"""

#Preparation: LOAD TRAINED MODEL FROM DIRECTORY
import gensim.models.keyedvectors as word2vec
skipg = word2vec.KeyedVectors.load_word2vec_format('DEOntologyTextMining/skipg.model')

#----------Algorithm 1: Domain words extraction
RT=[]
seeds=['sandbox','test-bench','testbed','pilot-plant','living-lab','pilot-line','pilot-facility','pilot-scale-plant','innovation-lab','innovation-hub','prototype-plant','prototyping-platform','open-innovation','facility','tool','test','experiment','hardware','software','federation','feasibility','vulnerability','stakeholder','user','prototype','creation','collaboration','participation','viability','virtual','modular','innovation','demonstration','heterogenous','heterogeneous','fidelity']
for seed in seeds:
    for x in skipg.wv.most_similar(seed,topn=500):
        if x[1]>0.5: #Threshold is set to 0.5 which is to find words with similarity >0.5, can be changed
            RT.append(x[0])
RT=set(RT)

#----------Algorithm 2:  Candidate concept and properties identification
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet as wn
from collections import defaultdict
from nltk import pos_tag
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
propertyList=[]
conceptList=[]
for word in RT:
    pos=get_wordnet_pos(word)
    if pos == 'n' or pos =='a': #if a word is noun or adjective
        conceptList.append(word) #add to concept list
    else:
        if pos == 'v': # if a word is verb
            propertyList.append(word) #add to property list
            
#----------Algorithm 3:  Ontology hierarchy building
# get vector
vectorList=[]
for word in conceptList:
    vectorList.append(skipg.wv.word_vec(word))

#-------3.1 Phase 1: get CF tree to obtain initial clustering
from sklearn.cluster import Birch
brc = Birch(branching_factor=180, n_clusters=None, threshold=2.5,compute_labels=True)
brc.fit(vectorList)
res=brc.predict(vectorList)
brc.subcluster_labels_

cluster1=[]
cluster2=[]
cluster3=[]
cluster4=[]
cluster5=[]
cluster6=[]
cluster7=[]
#tbc depends on how many clusters got

i=0
for label in res:
    if label==0:
        cluster1.append(conceptList[i])
    elif label==1:
        cluster2.append(conceptList[i])  
    elif label==2:
        cluster3.append(conceptList[i])
    elif label==3:
        cluster4.append(conceptList[i])
    elif label==4:
        cluster5.append(conceptList[i])  
    elif label==5:
        cluster6.append(conceptList[i])
    elif label==6:
        cluster7.append(conceptList[i])
    i=i+1   
#------3.2 Phase 2
from sklearn.cluster import AgglomerativeClustering   
#for each cluster do below:
vec=[]
for word in cluster1:
    vec.append(skipg.wv.word_vec(word))
model = AgglomerativeClustering(n_clusters=None,distance_threshold=4.4) #cluster1:3.4; cluster2:5.2; cluster3: 4; cluster4:3.6; cluster5:5.5;cluster7:5;cluster8:4.5;cluster9:5
model=model.fit(vec)
max(model.labels_)# this returns # of clusters -1
#print words and their subcluster number
for i in range(max(model.labels_)):
    j=0
    for lab in model.labels_:
        if lab==i:
            print(i,'->',cluster1[j])#this 'cluster1' needs to be changed everytime the working cluster changed
        j=j+1 
  
#plot dendrogem for visualisation
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    #R=dendrogram(linkage_matrix,**kwargs)
    R=dendrogram(linkage_matrix,show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.figure(figsize=(49, 36))
plot_dendrogram(model, labels=model.labels_)

plt.savefig('cluster1.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

#----------Algorithm 4:  Concept-property relation identification
# the following function only looks for top 1 similar concept, the rest can be obtained using wv.most_similar() function
propArr=[]
for prop in propertyList:
    sim=0
    tempsim=0
    prop_vec=skipg.wv.word_vec(prop)
    allocatedCon=''
    for con in conceptList:
        con_vec=skipg.wv.word_vec(con)
        tempsim=skipg.wv.similarity(prop,con)
        if tempsim>sim:
            sim=tempsim
            allocatedCon=con
    propArr.append(allocatedCon)
for i in range(len(propArr)):
    print(propertyList[i],'-> ',propArr[i])     
