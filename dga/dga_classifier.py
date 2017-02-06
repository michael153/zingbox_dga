"""Train and test bigram classifier"""

import os
import sys
import json
import numpy as np
import pandas as pd
from dataprocess import Dataprocess
from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.metrics import top_k_categorical_accuracy
from keras.models import model_from_json
from collections import defaultdict
import sklearn
from sklearn import feature_extraction
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split


data_dir = os.path.abspath('data')

def multi_model(max_features):
    model = Sequential()
    model.add(Dense(30, input_dim=max_features, init='uniform', activation='relu'))
    model.add(Dense(22, init='uniform', activation='relu'))                                               
    model.add(Dense(18, input_dim=max_features,init='uniform', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics = [top_k_categorical_accuracy])
    return model

def binary_model(max_features):
    model = Sequential()
    model.add(Dense(12, input_dim=max_features, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(2,input_dim=max_features, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',  metrics = ['accuracy'])
    return model

def main(type, num, max_epoch=50, nfolds=10, batch_size=128):
    """Run train/test on logistic regression model"""
    num = int(num)
    max_epoch = int(max_epoch)
    nfolds = int(nfolds)
    batch_size = int(batch_size)
    indata = Dataprocess(num).get_data(type,force=True)
    # Extract data and labels
    X = [x[1] for x in indata]
    X_length = [len(x) for x in X]
    labels = [x[0] for x in indata]
    label_set = list(set(labels))
    ngram_vectorizer = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2,3), min_df = 0.0001)
    countvec = ngram_vectorizer.fit_transform(X)
    countvec.toarray()
    countvec = np.append(countvec, [X_length], axis = 1)
    cols = ngram_vectorizer.get_feature_names()
    '''
    unknown_letter = defaultdict(int)
    for x in X:
        ngram_vectorizer = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2,3))
        countvec2 = ngram_vectorizer.fit_transform([x])
        cols2 = ngram_vectorizer.get_feature_names()
        print cols2
        for col in cols2:
            if cols2 not in cols1:
                unknown_letter[x] += 1
    
    print unknown_letter
    print unknown_letter.values()
    countvec['unknown'] = unknown_letter.values()
    '''
    max_features = countvec.shape[1]

    # Create feature vectors
    print "vectorizing data"
    thefile = open(type+'cols.txt', 'w')
    for item in cols:
        thefile.write("%s\n" % item)  

    # Convert labels to 0-18/0-2
    y = [label_set.index(x) for x in labels]
    #if type == 'multi':
    y = np_utils.to_categorical(y, len(label_set))

    final_data = []
    final_score = []
    best_m_auc = 0.0

    for fold in range(nfolds):
        print "fold %u/%u" % (fold+1, nfolds)
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(countvec, y,
                                                                           labels, test_size=0.2)
        print 'Build model...'
        if type == 'multi':
            model = multi_model(max_features)
        else:
            model = binary_model(max_features)
        print "Train..."
        X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05)
        print X_train.shape
        best_iter = -1
        best_auc = 0.0
        out_data = {}

        for ep in range(max_epoch):
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1)
            t_probs = model.predict_proba(X_holdout)
            t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)
            print 'Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc)
            if t_auc > best_auc:
                best_auc = t_auc
                best_iter = ep         
                #out_data = {'y': y_test, 'labels': label_test, 'probs':probs, 'epochs': ep}              
            else:
                if (ep-best_iter) >= 2:
                    break
        probs = model.predict_proba(X_test)
        m_auc = sklearn.metrics.roc_auc_score(y_test, probs)
        print 'score is %f' % m_auc
        if m_auc > best_m_auc:
            best_model = model
        
        final_test = [label_set[i] for i in y_test.argmax(axis=1)]
        final_prob = [label_set[i] for i in probs.argmax(axis=1)]
        top_prob = np.array([np.array(label_set)[i] for i in probs.argsort()])
        top_prob = np.concatenate((top_prob, np.array([label_test]).T), axis=1)
        metric = pd.DataFrame([final_test, final_prob])
        metric = metric.transpose()
        metric.columns = ['actual','pred']
        #metric = np.asarray(sklearn.metrics.confusion_matrix(final_test, final_prob))
        #print metric
        #metric = pd.DataFrame(data=metric, index=label_set, columns=label_set)
        top_prob_metric = pd.DataFrame(data=top_prob)
        metric.to_csv(os.path.join(data_dir,"final" + type + str(fold)+".csv"))
        top_prob_metric.to_csv(os.path.join(data_dir,"ranking" + type + str(fold)+".csv"))
        final_score.append(m_auc)
        print final_score
        model_json = best_model.to_json()
        with open(type+"_model.json", "w") as json_file:
            json_file.write(model_json)
            model_json = best_model.to_json()
        best_model.save_weights(type+"_model.h5") 

    return best_model

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])  
