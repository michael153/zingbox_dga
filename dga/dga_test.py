import os
import json
import numpy as np
import pandas as pd
from dataprocess import Dataprocess
<<<<<<< HEAD
from datagenerator import Datagenerator
=======
>>>>>>> master
from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.metrics import top_k_categorical_accuracy
from keras.models import model_from_json
import sklearn
from sklearn import feature_extraction
from sklearn.cross_validation import train_test_split
import urllib2
import tldextract

<<<<<<< HEAD
data_dir = os.path.abspath('data')
json_file = open('binary_model_json', 'r')
loaded_model_json = json_file.read()
json_file.close()
binary_model = model_from_json(loaded_model_json)
json_file = open('multi_model_json', 'r')
loaded_model_json = json_file.read()
json_file.close()
multi_model = model_from_json(loaded_model_json)
binary_model.load_weights("binary_model")
multi_model.load_weights("multi_model")
=======
data_dir = os.path.abspath('data/' + 'staging')
json_file = open('binary_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
binary_model = model_from_json(loaded_model_json)
json_file = open('multi_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
multi_model = model_from_json(loaded_model_json)

binary_model.load_weights("binary_model.h5")
multi_model.load_weights("multi_model.h5")
>>>>>>> master


cols1= [line.rstrip('\n') for line in open('binary'+'cols.txt')]
cols2= [line.rstrip('\n') for line in open('multi'+'cols.txt')]

def subtest(binary_model, multi_model, data, cols1, cols2):
<<<<<<< HEAD
    labels = ['Benign','Malicious']

    ngram_vectorizer = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2,3))
=======
    indata = Dataprocess(42000).get_data('binary')
    labels = list(set([x[0] for x in indata]))

    ngram_vectorizer = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2,3),min_df = 0.0001)
>>>>>>> master
    newvec = [0]*len(cols1)
    newcount = ngram_vectorizer.fit_transform(data)

    newcols = ngram_vectorizer.get_feature_names()
    for i in range(len(cols1)):
        if cols1[i] in newcols:
<<<<<<< HEAD
            newvec[i] = 1
    if len(newvec) > 20:
        newvec.append(1)
    else:
        newvec.append(0) 
    
    is_dga = [labels[i] for i in binary_model.predict_classes(np.array([newvec]))]
    binary_prob = binary_model.predict_proba(np.array([newvec]))[0]
    if sum(binary_prob) < 0.5:
        is_dga[0] = 'Benign'
    type_dga = None

    labels =  ['zeus', 'corebot', 'pushdo', 'ramnit', 'matsnu', 'banjori', 'tinba', 'rovnix', 'conficker', 'locky', 'cryptolocker']
 
    ngram_vectorizer = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2,3))    
    newvec = [0]*len(cols2)

=======
            newvec[i] = 1 
    is_dga = [labels[i] for i in binary_model.predict_classes(np.array([newvec]))]
    type_dga = None

    indata = Dataprocess(50000).get_data('multi')
    labels = list(set([x[0] for x in indata])) 
 
    ngram_vectorizer = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2,3))    
    newvec = [0]*len(cols2)
>>>>>>> master
    newcount = ngram_vectorizer.fit_transform(data)
    newcols = ngram_vectorizer.get_feature_names()
    for i in range(len(cols2)):
        if cols2[i] in newcols:
<<<<<<< HEAD
            newvec[i] = 1
    if len(newvec) > 20:
        newvec.append(1)
    else:
        newvec.append(0) 
    probs = multi_model.predict_proba(np.array([newvec]))

    top_prob = np.array([np.array(labels)[i] for i in probs.argsort()])
=======
            newvec[i] = 1 
    probs = multi_model.predict_proba(np.array([newvec]))

    top_prob = np.array([np.array(labels)[i] for i in probs.argsort()])
    print top_prob
>>>>>>> master
    top_prob = top_prob.tolist()[0]
    top_prob.reverse()
    type_dga = top_prob[:4]
    probs = probs.tolist()[0]
    probs = sorted(probs, reverse = True)
    probs = probs[:4]
<<<<<<< HEAD

    #print probs
    if is_dga[0] == 'Benign':
=======
    #print probs
    if is_dga == 'benign':
>>>>>>> master
        res = 'Safe'
    else:
        res = 'Malicious'
    print '%s Domain Address: %s   Top 4 Suspicious DGA Type: %s   Top 4 DGA Prob %s' % (res, data[0], ' '.join(type_dga), probs)
<<<<<<< HEAD
    return is_dga, type_dga, probs, binary_prob

def test(testdata, labels):
    is_dga_list = []
    type_dga_list = []
    probs_list = []
    binary_probs = []
    for d in testdata:     
        is_dga, type_dga, probs, binary_prob = subtest(binary_model, multi_model, [d], cols1, cols2)
        is_dga_list.append(is_dga[0])
        type_dga_list.append(type_dga)
        probs_list.append(probs)
        binary_probs.append(binary_prob)
         
    type_dga_list = pd.DataFrame(np.array(type_dga_list))
    probs_list = pd.DataFrame(np.array(probs_list))
    res = [testdata, labels, is_dga_list, binary_probs]
    res = pd.DataFrame(res).transpose()
    table = pd.concat([res, type_dga_list, probs_list], axis=1)
    table.columns = ['Domain','Label','Pred','Benign_prob','Top1','Top2','Top3','Top4','Prob1','Prob2','Prob3','Prob4',]
    print table
    return table

=======
    return is_dga, type_dga, probs
>>>>>>> master

domain_list = []
with open('dga.txt','r') as f:
    for line in f:
        address = line.split(' ')[1]
        domain = tldextract.extract(address).domain
        domain_list.append(domain)
<<<<<<< HEAD
print domain_list

labels = ['conficker']*len(domain_list)
table = test(domain_list, labels)
table.to_csv(os.path.join(data_dir,'res_'+'cryptolocker'+'.csv'))
indata = Datagenerator(43000,43300).get_data(force=True)

X = [x[1] for x in indata if len(x[1]) > 1]
print len(X)
labels = [x[0] for x in indata if len(x[1]) > 1]
table = test(X, labels)
table.to_csv(os.path.join(data_dir,'res_'+'all_test'+'.csv'))
=======


'''

s_dga, type_dga = subtest(binary_model, multi_model, ['csteifhtzsjqil'], cols1, cols2)
s_dga, type_dga = subtest(binary_model, multi_model, ['xwzhxswxhk'], cols1, cols2)
s_dga, type_dga = subtest(binary_model, multi_model, ['noxgcxxvrlhywr'], cols1, cols2)
'''
labels = ['Malicious']*len(domain_list)
is_dga_list = []
type_dga_list = []
probs_list = []
for d in domain_list[:500]:
    is_dga, type_dga, probs = subtest(binary_model, multi_model, [d], cols1, cols2)
    is_dga_list.append(is_dga[0])
    type_dga_list.append(type_dga)
    probs_list.append(probs)

res = [labels, is_dga_list, type_dga_list, probs_list, domain_list]
table = pd.DataFrame(res).transpose()
table.columns = ['true','pred1','pred2','prob','address']
print table
table.to_csv(os.path.join(data_dir,'test_res.csv'))
>>>>>>> master
