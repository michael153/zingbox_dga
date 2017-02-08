import os
import json
import numpy as np
import pandas as pd
from dataprocess import Dataprocess
from datagenerator import Datagenerator
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


cols1= [line.rstrip('\n') for line in open('binary'+'cols.txt')]
cols2= [line.rstrip('\n') for line in open('multi'+'cols.txt')]

def subtest(binary_model, multi_model, data, cols1, cols2):
    labels = ['Benign','Malicious']

    ngram_vectorizer = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2,3),min_df = 0.0001)
    newvec = [0]*len(cols1)
    newcount = ngram_vectorizer.fit_transform(data)

    newcols = ngram_vectorizer.get_feature_names()
    for i in range(len(cols1)):
        if cols1[i] in newcols:
            newvec[i] = 1
    if len(newvec) > 20:
        newvec.append(1)
    else:
        newvec.append(0) 
    
    is_dga = [labels[i] for i in binary_model.predict_classes(np.array([newvec]))]
    type_dga = None

    labels =  ['corebot', 'zeus', 'pushdo', 'matsnu', 'banjori', 'tinba', 'rovnix', 'conficker']
 
    ngram_vectorizer = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2,3))    
    newvec = [0]*len(cols2)

    newcount = ngram_vectorizer.fit_transform(data)
    newcols = ngram_vectorizer.get_feature_names()
    for i in range(len(cols2)):
        if cols2[i] in newcols:
            newvec[i] = 1
    if len(newvec) > 20:
        newvec.append(1)
    else:
        newvec.append(0) 
    probs = multi_model.predict_proba(np.array([newvec]))

    top_prob = np.array([np.array(labels)[i] for i in probs.argsort()])
    top_prob = top_prob.tolist()[0]
    top_prob.reverse()
    type_dga = top_prob[:4]
    probs = probs.tolist()[0]
    probs = sorted(probs, reverse = True)
    probs = probs[:4]
    #print probs
    if is_dga == ['benign']:
        res = 'Safe'
    else:
        res = 'Malicious'
    print '%s Domain Address: %s   Top 4 Suspicious DGA Type: %s   Top 4 DGA Prob %s' % (res, data[0], ' '.join(type_dga), probs)
    return is_dga, type_dga, probs

def test(testdata, labels):
    is_dga_list = []
    type_dga_list = []
    probs_list = []
    for d in testdata:
        try:
            is_dga, type_dga, probs = subtest(binary_model, multi_model, [d], cols1, cols2)
        except:
            pass
        is_dga_list.append(is_dga[0])
        type_dga_list.append(type_dga)
        probs_list.append(probs)
    type_dga_list = pd.DataFrame(np.array(type_dga_list))
    probs_list = pd.DataFrame(np.array(probs_list))
    res = [testdata, labels, is_dga_list]
    res = pd.DataFrame(res).transpose()
    table = pd.concat([res, type_dga_list, probs_list], axis=1)
    table.columns = ['Domain','Label','Pred','Top1','Top2','Top3','Top4','Prob1','Prob2','Prob3','Prob4',]
    print table
    return table


domain_list = []
with open('dga.txt','r') as f:
    for line in f:
        address = line.split(' ')[1]
        domain = tldextract.extract(address).domain
        domain_list.append(domain)

labels = ['cryptolocker']*len(domain_list)
table = test(domain_list, labels)
table.to_csv(os.path.join(data_dir,'res_'+'cryptolocker'+'.csv'))
indata = Datagenerator(40000,40020).get_data(force=True)

X = [x[1] for x in indata]
print len(X)
labels = [x[0] for x in indata]
table = test(X, labels)
table.to_csv(os.path.join(data_dir,'res_'+'all_test'+'.csv'))