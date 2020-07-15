"""An example program that uses the elsapy module"""

from elsapy.elsclient import ElsClient
from elsapy.elsprofile import ElsAuthor, ElsAffil
from elsapy.elsdoc import FullDoc, AbsDoc
from elsapy.elssearch import ElsSearch
import json
    
## Load configuration
con_file = open("config.json")
config = json.load(con_file)
con_file.close()

## Initialize client
client = ElsClient(config['apikey'])
client.inst_token = config['insttoken']

## Author example
# Initialize author with uri
my_auth = ElsAuthor(
        uri = 'https://api.elsevier.com/content/author/author_id/7004367821')
# Read author data, then write to disk
if my_auth.read(client):
    print ("my_auth.full_name: ", my_auth.full_name)
    my_auth.write()
else:
    print ("Read author failed.")

## Affiliation example
# Initialize affiliation with ID as string
my_aff = ElsAffil(affil_id = '60101411')
if my_aff.read(client):
    print ("my_aff.name: ", my_aff.name)
    my_aff.write()
else:
    print ("Read affiliation failed.")

## Scopus (Abtract) document example
# Initialize document with ID as integer
scp_doc = AbsDoc(scp_id = 84872135457)
if scp_doc.read(client):
    print ("scp_doc.title: ", scp_doc.title)
    scp_doc.write()   
else:
    print ("Read document failed.")

## ScienceDirect (full-text) document example using PII
pii_doc = FullDoc(sd_pii = 'S1674927814000082')
if pii_doc.read(client):
    print ("pii_doc.title: ", pii_doc.title)
    pii_doc.write()   
else:
    print ("Read document failed.")

## ScienceDirect (full-text) document example using DOI
doi_doc = FullDoc(doi = '10.1016/S1525-1578(10)60571-5')
if doi_doc.read(client):
    print ("doi_doc.title: ", doi_doc.title)
    doi_doc.write()   
else:
    print ("Read document failed.")


## Load list of documents from the API into affilation and author objects.
# Since a document list is retrieved for 25 entries at a time, this is
#  a potentially lenghty operation - hence the prompt.
print ("Load documents (Y/N)?")
s = input('--> ')

if (s == "y" or s == "Y"):

    ## Read all documents for example author, then write to disk
    if my_auth.read_docs(client):
        print ("my_auth.doc_list has " + str(len(my_auth.doc_list)) + " items.")
        my_auth.write_docs()
    else:
        print ("Read docs for author failed.")

    ## Read all documents for example affiliation, then write to disk
    if my_aff.read_docs(client):
        print ("my_aff.doc_list has " + str(len(my_aff.doc_list)) + " items.")
        my_aff.write_docs()
    else:
        print ("Read docs for affiliation failed.")

## Initialize author search object and execute search
auth_srch = ElsSearch('authlast(keuskamp)','author')
auth_srch.execute(client)
print ("auth_srch has", len(auth_srch.results), "results.")

## Initialize affiliation search object and execute search
aff_srch = ElsSearch('affil(amsterdam)','affiliation')
aff_srch.execute(client)
print ("aff_srch has", len(aff_srch.results), "results.")

## Initialize doc search object using Scopus and execute search, retrieving 
#   all results
doc_srch = ElsSearch("AFFIL(dartmouth) AND AUTHOR-NAME(lewis) AND PUBYEAR > 2011",'scopus')
doc_srch.execute(client, get_all = True)
print ("doc_srch has", len(doc_srch.results), "results.")

## Initialize doc search object using ScienceDirect and execute search, 
#   retrieving all results
doc_srch = ElsSearch("star trek vs star wars",'sciencedirect')
doc_srch.execute(client, get_all = False)
print ("doc_srch has", len(doc_srch.results), "results.")




#read doi csv
import csv
doiList = []
csvReader = csv.reader(open('doi.csv', newline='',encoding='utf-8-sig'), delimiter=' ', quotechar='|')
for row in csvReader:
    #print(','.join(row))
    doiList.append(','.join(row))



#loop reading
download=[]
for i in doiList:
    doi_doc = FullDoc(doi = i)
    if doi_doc.read(client):
        #print ("doi_doc.title: ", doi_doc.title)
        doi_doc.write()  
        download.append('True')
        
    else:
        print ("Read document failed :")
        #print ("doi_doc.title: ", doi_doc.title)
        download.append('False')
j=0

for i in doiList:
    if j>213:
        doi_doc = FullDoc(doi = i)
        if doi_doc.read(client):
        #print ("doi_doc.title: ", doi_doc.title)
            doi_doc.write()  
        
        else:
            print ("Read document failed :")
            print(i) 
            #print ("doi_doc.title: ", doi_doc.title)
    j=j+1;
    
        
#hardcode part
#93       
json_file=open("data/https%3A%2F%2Fapi.elsevier.com%2Fcontent%2Farticle%2Fdoi%2F10.1016%2Fj.nima.2019.01.071.json")
data = json.load(json_file)
textList[93]=data['originalText']
#59
json_file=open("data/https%3A%2F%2Fapi.elsevier.com%2Fcontent%2Farticle%2Fdoi%2F10.1016%2Fj.ijepes.2018.07.058.json")
data = json.load(json_file)
textList[59]=data['originalText']


#NOT IN USE END

import os, json
import pandas as pd

#--------------------READ JSON FILE
textList=[]
#idList=[]
path_to_json = 'data/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
for file in json_files:
    json_file=open('data/'+file) #read json file stored in data folder
    data = json.load(json_file)
    textList.append(data['originalText'])
    #idList.append(data['scopus-eid'])
    #print(data[''])

#----------------Read text file
textList2=[]
path_to_txt = 'pdfSciDirect'
txt_files = [pos_txt for pos_txt in os.listdir(path_to_txt) if pos_txt.endswith('.txt')]
for file in txt_files:
    txt_file=open('pdfSciDirect/'+file) #read json file stored in data folder
    textList2.append(txt_file.read())
#debug list item type
i = 0
count=0
textListAvi=[] #json file with valid contents inside
while i < len(textList):
    #print(type(textList[i]))
    if not isinstance(textList[i],str):
        print(i)
        count=count+1
    #else:
        #textListAvi.append(textList2[i])
    i=i+1

finalList=textList+textList2+textList3  #all the textList combined together
    
#Text pre-processing
textListProcessed = finalList
#---step1: transform to lower case
for i in range(len(textListProcessed)):
    textListProcessed[i] = textListProcessed[i].lower()
#----step2: replace keyword
import re
for i in range(len(textListProcessed)):
    textListProcessed[i] = textListProcessed[i].replace("\n"," ")
    textListProcessed[i] = textListProcessed[i].replace("- ","")
    textListProcessed[i] = textListProcessed[i].replace("test bed","testbed")
    textListProcessed[i] = textListProcessed[i].replace("testbeds","testbed")
    textListProcessed[i] = textListProcessed[i].replace("pilot plant","pilot-plant")
    textListProcessed[i] = textListProcessed[i].replace("pilot-plant","pilot-plant")
    textListProcessed[i] = textListProcessed[i].replace("pilot_plants","pilot-plant")
    textListProcessed[i] = textListProcessed[i].replace("living lab","living-lab")
    textListProcessed[i] = textListProcessed[i].replace("living laboratory","living-lab")
    textListProcessed[i] = textListProcessed[i].replace("living laboratories","living-lab")
    textListProcessed[i] = textListProcessed[i].replace("living-lab","living-lab")
    textListProcessed[i] = textListProcessed[i].replace("living_labs","living-lab")
    textListProcessed[i] = textListProcessed[i].replace("innovation lab","innovation-lab")
    textListProcessed[i] = textListProcessed[i].replace("innovation hub","innovation-hub")
    textListProcessed[i] = textListProcessed[i].replace("innovation space","innovation-space")
    textListProcessed[i] = textListProcessed[i].replace("innovation center","innovation-center")
    textListProcessed[i] = textListProcessed[i].replace("innovation centre","innovation-center")
    textListProcessed[i] = textListProcessed[i].replace("pilot scale plant","pilot-scale-plant")
    textListProcessed[i] = textListProcessed[i].replace("pilot-scale plant","pilot-scale-plant")
    textListProcessed[i] = textListProcessed[i].replace("pilot line","pilot-line")
    textListProcessed[i] = textListProcessed[i].replace("pilot facility","pilot-facility")
    textListProcessed[i] = textListProcessed[i].replace("pilot facilities","pilot-facility")
    textListProcessed[i] = textListProcessed[i].replace("pilot lab","pilot-lab")
    textListProcessed[i] = textListProcessed[i].replace("pilot laboratory","pilot-lab")
    textListProcessed[i] = textListProcessed[i].replace("pilot demonstration","pilot-demonstration")
    textListProcessed[i] = textListProcessed[i].replace("prototype platform","prototyping-platform")
    textListProcessed[i] = textListProcessed[i].replace("prototype platforms","prototyping-platform")
    textListProcessed[i] = textListProcessed[i].replace("prototyping platform","prototyping-platform")
    textListProcessed[i] = textListProcessed[i].replace("prototyping platforms","prototyping-platform")
    textListProcessed[i] = textListProcessed[i].replace("prototype plant","prototype-plant")
    textListProcessed[i] = textListProcessed[i].replace("prototype plants","prototype-plant")
    textListProcessed[i] = textListProcessed[i].replace("demonstration facility","demonstration-facility")
    textListProcessed[i] = textListProcessed[i].replace("demonstration facilities","demonstration-facility")
    textListProcessed[i] = textListProcessed[i].replace("demonstration center","demonstration-center")
    textListProcessed[i] = textListProcessed[i].replace("demonstration centre","demonstration-center")
    textListProcessed[i] = textListProcessed[i].replace("open innovation","open-innovation")
    textListProcessed[i] = textListProcessed[i].replace("scaleup","scale-up")
    textListProcessed[i] = textListProcessed[i].replace("scale up","scale-up")
    textListProcessed[i] = textListProcessed[i].replace("field laboratory","field-lab")
    textListProcessed[i] = textListProcessed[i].replace("field lab","field-lab")
    textListProcessed[i] = textListProcessed[i].replace("test bench","test-bench")
    textListProcessed[i] = textListProcessed[i].replace("urban laboratory","urban-lab")
    textListProcessed[i] = textListProcessed[i].replace("urban lab","urban-lab")
    textListProcessed[i] = textListProcessed[i].replace("city laboratory","city-lab")
    textListProcessed[i] = textListProcessed[i].replace("city lab","city-lab")
    textListProcessed[i] = textListProcessed[i].replace("maker space","maker-space")
    textListProcessed[i] = textListProcessed[i].replace("test bench","test-bench")
    textListProcessed[i] = textListProcessed[i].replace("test benches","test-bench")    
    textListProcessed[i] = textListProcessed[i].replace("demonstration plant","demonstration-plant") 
    textListProcessed[i] = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', textListProcessed[i])
#....to be updated after keyword occurrence analysis
#------to test the result of step 2
new_text=""
new_text=" ".join(w for w in ProcessList2[7])
f = open("demotext.txt", "a")
f.write(new_text)
f.close()
#--end of test

#----step 3:tokinization
#from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
for i in range(len(textListProcessed)):
    textListProcessed[i]=word_tokenize(textListProcessed[i])
   
#----step 4:remove stopwords, digits, and words with length < 4 and length > 25
#build stopword table
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
#build english word check table
#import re
#english_check = re.compile(r'[a-z]')

from nltk.corpus import wordnet
ProcessList2=textListProcessed
for i in range(len(textListProcessed)):
    filtered_sent=[]
    for w in textListProcessed[i]:
        if w not in stop_words and len(w)>4 and len(w)<25 and (wordnet.synsets(w) or w =='testbed' or w=='test-bench' or w=='maker-space' or w=='urban-lab'or w=='city-lab'or w=='field-lab'or w=='living-lab' or w=='pilot-plant' or w=='test-bench' or w=='sandbox' or w=='pilot-line' or w=='innovation-lab' or w=='innovation-hub' or w=='innovation-space' or w=='innovation-center' or w=='pilot-scale-plant' or w=='pilot-line' or w=='pilot-facility' or w=='pilot-lab' or w=='pilot-demonstration' or w=='prototyping-platform' or w=='prototype-plant' or w=='demonstration-facility' or w=='demonstration-center' or w=='open-innovation' or w=='scale-up'):
            filtered_sent.append(w)
        ProcessList2[i]=filtered_sent
#------not in use:porter stem
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
for i in range(len(ProcessList2)):
    stemmed_words=[]
    for w in ProcessList2[i]:
        stemmed_words.append(ps.stem(w))
#------------------------------

#---step 5: lemmentization based on POS tag
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet as wn
from collections import defaultdict
from nltk import pos_tag
#build dictionary
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
lemmatizer = WordNetLemmatizer() 
for i in range(len(ProcessList2)):
    lemmatize_words=[]
    for token, tag in pos_tag(ProcessList2[i]):
        lemma = lemmatizer.lemmatize(token, tag_map[tag[0]])
        lemmatize_words.append(lemma)
    ProcessList2[i]=lemmatize_words
#--needs further pruning----
        
#for small unit testing-------      
for token, tag in pos_tag(ProcessList2[0]):
    lemma = lemmatizer.lemmatize(token, tag_map[tag[0]])
    print(token, "=>", lemma)
#end testing
    
#word2vec training
import gensim 
from gensim.models import Word2Vec 
#create CBOW model
cbow = Word2Vec(ProcessList2, min_count=15,window=5)
skipg = Word2Vec(ProcessList2, sg=1, min_count=20,window=5,size=150)

#save
skipg4.save("new_skipg4.model")
model.wv.save_word2vec_format('bigram_model_size150_min15.model')
words = list(cbow.wv.vocab)

from gensim.models import Phrases
from gensim.models.phrases import Phraser
bigram = Phrases(ProcessList2[0], min_count=5, threshold=10)
bigram_mod = gensim.models.phrases.Phraser(bigram) #phraser
train_bigram_model = Word2Vec(bigram_token, size=150, min_count=15)
bigram_token = []
for sent in textList[0]:
    print(bigram_mod[sent])
    #bigram_token.append(bigram_mod[sent])

dict = gensim.corpora.Dictionary(bigram_token)
print(dict.token2id)
#Convert the word into vector, and now you can use tfidf model from gensim 
corpus = [dict.doc2bow(text) for text in bigram_token]
tfidf_model = models.TfidfModel(corpus)

for unit in textListProcessed:                
            bigram_sentence = u' '.join(bigram[unit])


X = cbow[cblow.wv.vocab]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
result = pca.fit_transform(X)
import matplotlib.pyplot as pyplot
#f = pyplot.figure()
pyplot.scatter(result[:, 0], result[:, 1])
words = list(cbow.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
#pyplot.xlim(1.3, 4.0)
#f.savefig("foo.pdf", bbox_inches='tight')
pyplot.show()

#tf-idf
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([documentA, documentB])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)

#-----tsne visualization
def display_closestwords_tsnescatterplot(model, word, size):
    
    arr = np.empty((0,size), dtype='f')
    word_labels = ['testbed']
    close_words = model.similar_by_word(word)
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
#---------------

#---find string---
count=0
for i in range(len(ProcessList2)):
    for word in ProcessList2[i]:
        if(word.find('pilotplant')):
            count+=1

#---clean sentence----
import re
def clean_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r’[^a-z0-9\s]’, '’, sentence)
    return re.sub(r’\s{2,}’, ' ', sentence)




#consolidated form
import nltk
from nltk.corpus import stopwords
words = set(nltk.corpus.words.words())
stop_words = stopwords.words('english')


file_name = 'Full path to your file'
with open(file_name, 'r') as f:
    text = f.read()
    text = text.replace('\n', ' ')

new_text = " ".join(w for w in nltk.wordpunct_tokenize(text)
                    if w.lower() in words
                    and w.lower() not in stop_words
                    and len(w.lower()) > 1)

print(new_text)
##-----------------end---------
import string
def isEnglish(s):
    return s.translate(None, string.punctuation).isalnum()

new_text=" ".join(w for w in textListProcessed[0] if len(w)>1 and w not in stop_words and english_check.match(w))





#bigram testing

docs = ['new york is is united states', 'new york is most populated city in the world','i love to stay in new york']

token_ = [doc.split(" ") for doc in docs]
bigram = Phrases(token_, min_count=1, threshold=2,delimiter=b' ')


bigram_phraser = Phraser(bigram)

bigram_token = []
for sent in token_:
    bigram_token.append(bigram_phraser[sent])

model = Word2Vec(min_count=1)
model.build_vocab(bigram_token)
model.train(bigram_token,total_examples=model.corpus_count,epochs=model.epochs)


#calculate total # of words
unique_words=[]
total_count=0
for doc in ProcessList2:
    total_count=total_count+len(doc)
    for word in doc:
        unique_words.append(word)
print(len(set(unique_words)))
    
    
    
min_count=[100,110,120,130,140,150,160,170,180,190,200]
window=[5,6,7,8]
iter_set=[10,15,20,25,30,35,40,45,50]
ns_exponent=0.75
hs=[0,1]
loss=[]
i=0
for count in min_count:
    for win in window:
        for hs_val in hs:
            for iteration in iter_set:
                string='cbow_min'+str(count)+'_win'+str(win)+'_iter'+str(iteration)+'_hs'+str(hs_val)+'_ns0.75.model'
                if i==90:
                    print(string)
                #cbow = Word2Vec(ProcessList2, min_count=count,window=win,ns_exponent=ns_exponent,iter=iteration,hs=hs_val,compute_loss=True)
                #loss.append(cbow.get_latest_training_loss())
                #cbow.wv.save_word2vec_format(string)
                i=i+1
            




