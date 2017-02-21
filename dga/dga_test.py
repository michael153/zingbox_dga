import os
import json
import sys
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

    newvec = [0]*len(cols1)
    for i in range(len(cols1)):
        if cols1[i] in data[0]:
            newvec[i] = 1
    if len(data[0]) > 15:
        newvec.append(1)
    else:
        newvec.append(0) 
    is_dga = [labels[i] for i in binary_model.predict_classes(np.array([newvec]))]
    binary_prob = binary_model.predict_proba(np.array([newvec]))[0]
    type_dga = None  

    labels =   ['zeus', 'corebot', 'goz', 'pushdo', 'ramnit', 'matsnu', 'banjori', 'tinba', 'rovnix', 'conficker', 'locky', 'cryptolocker']    
    newvec = [0]*len(cols2)
    for i in range(len(cols2)):
        if cols2[i] in data[0]:
            newvec[i] = 1
    if len(data[0]) > 15:
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
    if is_dga[0] == 'Benign':
        res = 'Safe'
    else:
        res = 'Malicious'
    print '%s Domain Address: %s   Top 4 Suspicious DGA Type: %s   Top 4 DGA Prob %s' % (res, data[0], ' '.join(type_dga), probs)
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

def main(testcase, domain = True):   
    domain_list = []
    fp = open(os.path.abspath('tests')+"/java_test.txt", "w")
    with open(os.path.abspath('tests')+'/'+testcase,'r') as f:
        for line in f:
            address = line
            if domain:
                domain = tldextract.extract(address).domain
            else:
                domain = tldextract.extract(address).subdomain
            fp.write(domain + '\n')
            domain_list.append(domain)
    fp.close()
    labels = ['conficker']*len(domain_list)
    table = test(domain_list, labels)
    table.to_csv(os.path.join(data_dir,'test'+'_output'+'.csv'))

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])
