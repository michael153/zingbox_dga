import os
import sys
import json
import time
import sklearn
import numpy as np
import pandas as pd
import commands

from numpy import array
from scipy.sparse import csr_matrix
from dataprocess import Dataprocess
from keras.layers.core import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.metrics import top_k_categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import Callback
from collections import defaultdict
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split


data_dir = os.path.abspath('data')

class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.000005, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print("***Warning: Early stopping requires %s available!" % self.monitor)
        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

def build_model(maxlen, num_features, dim_output, config):
    model = Sequential()
    layers = {'hidden1': 64, 'hidden2': 64, 'hidden3': 64, 'output': dim_output}

    model.add(Embedding(num_features, config["input"]["num"], input_length=maxlen))

    print colors.WARNING

    l = config["hidden_layers"].__len__()
    for i in range(0, l):
        k = config["hidden_layers"][i]
        t = k["type"]
        if t == "LSTM":
            model.add(LSTM(k["num"], return_sequences=(True if k["return_sequences"] == "True" else False)))
            print "Added LSTM(" + str(k["num"]) + ", return_sequences=" + k["return_sequences"] + ")"
            if "dropout" in k:
                model.add(Dropout(k["dropout"]))
                print "Added Dropout(" + str(k["dropout"]) + ")"
        elif t in ["relu", "sigmoid", "tanh", "softmax"]:
            model.add(Dense(k["num"], init='uniform', activation=t))
            print "Added Dense(" + str(k["num"]) + ", init='uniform', activation=" + t + ")"

    # model.add(LSTM(layers['hidden1'], return_sequences=True))
    # model.add(Dropout(0.1))

    # model.add(LSTM(layers['hidden2'],return_sequences=True))
    # model.add(Dropout(0.2))

    # model.add(LSTM(layers['hidden3'],return_sequences=False))
    # model.add(Dropout(0.2))

    # model.add(Dense(150, init='uniform', activation='relu'))

    model.add(Dense(output_dim=dim_output, init='uniform', activation=config["output"]["type"]))
    print "Added Dense(output_dim=" + str(dim_output) + ", init='uniform', activation=" + config["output"]["type"] + ")"
    print colors.ENDC

    start = time.time()
    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=config["metrics"])
    print "Compilation Time : ", time.time() - start
    return model

def csc_vappend(countvec, list):
    countvec_row = []
    nrow = countvec.shape[0]
    ncol = countvec.shape[1]
    for i in range(len(countvec.indptr)-1):
        countvec_row.extend([i]*(countvec.indptr[i+1] - countvec.indptr[i]))
    new_row = countvec_row + range(countvec.shape[0])
    new_col = np.array(countvec.indices.tolist()+ [countvec.shape[1]]*countvec.shape[0])
    new_data = np.array(countvec.data.tolist() + list)
    countvec = csr_matrix((new_data,(new_row, new_col)), shape=(nrow, ncol+1))
    return countvec

def main(num, max_epoch=50, nfolds=10, batch_size=128):
    """Run train/test on logistic regression model"""
    num = int(num)
    max_epoch = int(max_epoch)
    nfolds = int(nfolds)
    batch_size = int(batch_size)
    indata = Dataprocess(num).get_data("multi",force=True)
    # Extract data and labels
    X = [x[1] for x in indata]
    labels = [x[0] for x in indata]

    valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}
    valid_labels = {x:idx for idx, x in enumerate(set(labels))}

    # num_saved = int(open('rnn/num_saved.txt').read().rstrip('\n'))
    # num_saved += 1
    num_saved = int(time.time())

    open("rnn/multi_rnn_valid_chars.json", 'w').write(json.dumps(valid_chars, indent=4))
    open("rnn/multi_rnn_valid_labels.json", 'w').write(json.dumps(valid_labels, indent=4))

    commands.getoutput('mkdir rnn/saves/rnn_%s' % str(num_saved))
    open("rnn/saves/rnn_%s/multi_rnn_valid_chars.json" % str(num_saved), 'w').write(json.dumps(valid_chars, indent=4))
    open("rnn/saves/rnn_%s/multi_rnn_valid_labels.json" % str(num_saved), 'w').write(json.dumps(valid_labels, indent=4))

    max_features = len(valid_chars) + 1
    maxlen = np.max([len(x) for x in X])

    # print(colors.FAIL + "len(maxlen): " + str(maxlen) + colors.ENDC)

    open("rnn/max_len.txt", 'w').write(str(maxlen) + "\n")
    open("rnn/saves/rnn_%s/max_len.txt" % str(num_saved), 'w').write(str(maxlen) + "\n")

    # print "len(max_features): " + str(max_features)

    # Convert characters to int and pad
    X = [[valid_chars[y] for y in x] for x in X]
    X = sequence.pad_sequences(X, maxlen=maxlen)
    
    # print str(X.shape)
    # Add Length Feature
    # countvec = csc_vappend(countvec, X_length)
    
    # Create feature vectors
    print "Vectorizing data..." 

    # Convert labels to 0-12/0-2
    y = [valid_labels[x] for x in labels]
    y = np_utils.to_categorical(y, len(valid_labels))
    # print "y:%s" % y
    final_data = []
    final_score = []
    best_m_auc = 0.0

    callbacks = [
        EarlyStoppingByLossVal(monitor='val_loss', value=0.00001, verbose=1),
    ]

    # print colors.OKGREEN
    # print X.shape
    # print y.shape
    # print labels.__len__()
    # print colors.ENDC

    print colors.WARNING + "Reading config file..." + colors.ENDC
    cur_config = json.load(open("rnn/cur_config.json"))
    cur_config["output"]["num"] = len(valid_labels)
    open("rnn/saves/rnn_%s/cur_config.json" % str(num_saved), 'w').write(json.dumps(cur_config, indent=4))
    
    for fold in range(nfolds):
        print "Fold %u/%u" % (fold+1, nfolds)
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels, test_size = .2)
        # print "Input shape: " + str(X_train.todense().shape)
        # print "Labels shape: " + str(y_train.shape) + "\n"
        print "Input shape: " + str(X_train.shape)

        print 'Build model...'
        print colors.OKGREEN + "maxlen=" + str(maxlen) + "\nnum_features=" + str(X.__len__()) + "\nlen(valid_labels)=" + str(len(valid_labels)) + colors.ENDC
        
        model = build_model(maxlen, X.__len__(), len(valid_labels), cur_config)
        

        X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05)
        best_iter = -1
        best_auc = 0.0
        out_data = {}


        for ep in range(max_epoch):
            # model.fit(X_train.todense(), y_train, batch_size=batch_size, nb_epoch=1, callbacks=callbacks)
            # print colors.FAIL
            # print "len(X_train): " + str(len(X_train))
            # print "shape(X_train): " + str(X_train.shape)
            # print "len(y_train): " + str(len(y_train))
            # print "shape(y_train): " + str(y_train.shape)
            # print colors.ENDC

            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, validation_split=0.05, callbacks=callbacks)
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
        print '\nScore is %f' % m_auc
        if m_auc > best_m_auc:
            best_model = model
        
        # Save prediction result
        final_test = [labels[i] for i in y_test.argmax(axis=1)]
        final_prob = [labels[i] for i in probs.argmax(axis=1)]
        top_prob = np.array([np.array(labels)[i] for i in probs.argsort()])
        top_prob = np.concatenate((top_prob, np.array([label_test]).T), axis=1)
        output = pd.DataFrame([final_test, final_prob])
        output = output.transpose()
        output.columns = ['actual','pred']
        top_prob_output = pd.DataFrame(data=top_prob)
        output.to_csv(os.path.join(data_dir,"final" + "multi_RNN_" + str(fold)+".csv"))
        top_prob_output.to_csv(os.path.join(data_dir,"ranking" + "multi_RNN_" + str(fold)+".csv"))
        final_score.append(m_auc)
        
        print final_score
        print "CLASSIFIER AVERAGE FINAL SCORE: %s" % str(sum(final_score)/len(final_score))
        # Save Model
        
        best_model.save_weights("rnn/multi_RNN"+"_model") 
        best_model.save_weights("rnn/saves/rnn_%s/multi_RNN_model" % str(num_saved))
        

        model_json = best_model.to_json()

        json_file = open("rnn/multi_RNN"+"_model_json", "w")
        json_file.write(model_json)
        json_file.close()
    
        json_saved_file = open("rnn/saves/rnn_%s/multi_RNN_model_json" % str(num_saved), 'w')
        json_saved_file.write(model_json)
        json_saved_file.close()
        open('rnn/num_saved.txt', 'w').write(str(num_saved))

    return best_model

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])