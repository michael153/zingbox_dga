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
from keras.preprocessing import sequence
import sklearn
from sklearn import feature_extraction
from sklearn.cross_validation import train_test_split
import urllib2
import tldextract

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

data_dir = os.path.abspath('data')

json_file = open('binary_model_json', 'r')
loaded_model_json = json_file.read()
json_file.close()
binary_model = model_from_json(loaded_model_json)

json_file = open('multi_model_json', 'r')
loaded_model_json = json_file.read()
json_file.close()
multi_model = model_from_json(loaded_model_json)

json_file = open('rnn/multi_RNN_model_json', 'r')
loaded_model_json = json_file.read()
json_file.close()
multi_RNN_model = model_from_json(loaded_model_json)

binary_model.load_weights("binary_model")
multi_model.load_weights("multi_model")
multi_RNN_model.load_weights("rnn/multi_RNN_model")

cols1= [line.rstrip('\n') for line in open('binary'+'cols.txt')]
cols2= [line.rstrip('\n') for line in open('multi'+'cols.txt')]
valid_chars = json.load(open('rnn/multi_rnn_'+'valid_chars.json'))
valid_labels = json.load(open('rnn/multi_rnn_'+'valid_labels.json'))
maxlen = int(open('rnn/max_len.txt').read().rstrip('\n'))

def init(file_name):
	domain_list = []
	for line in open(file_name):
		address = line
		e = (tldextract.extract(address.rstrip('\n')).domain if tldextract.extract(address.rstrip('\n')).domain.__len__() > tldextract.extract(address.rstrip('\n')).subdomain.__len__() else tldextract.extract(address.rstrip('\n')).subdomain)
		if e != "":
			domain_list.append(e)
	return domain_list
	# e = ""
	# if "," in address:
	# 	e = tldextract.extract(line.rstrip('\n').strip(',')).subdomain
	# else:
	# 	e = tldextract.extract(address).subdomain
	# print "e: " + str(e)
	# if e != "":
	# 	domain_list.append(e)

# domain = domain_list[0]
# print(str(domain_list))

labels =   ['zeus', 'corebot', 'goz', 'pushdo', 'ramnit', 'matsnu', 'banjori', 'tinba', 'rovnix', 'conficker', 'locky', 'cryptolocker']
model_type = ''

def test_multi(file_name):
	domain_list = init(file_name)
	total = 0 
	conficker = 0
	bad = []
	print bcolors.FAIL + str(domain_list.__len__()) + " Domains...\n" + str(len(cols2)) + " cols(2)...\n" + bcolors.ENDC
	for domain in domain_list:
		newvec = [0]*len(cols2)
		for i in range(len(cols2)):
			if cols2[i] in domain:
				newvec[i] = 1
		# if len(domain) > 15:
		# 	newvec.append(1)
		# else:
		# 	newvec.append(0)
		probs = multi_model.predict_proba(np.array([newvec]), verbose=0)
		total += 1
		
		if np.amax(probs) == probs[0][9]:
			conficker += 1
		else:
			bad.append(domain)
		
		failed_str = "Failed = (" + str(total - conficker) + ", " + str(float(total - conficker)/total) + ")"
		colored_str = (bcolors.OKGREEN if np.amax(probs) == probs[0][9] else bcolors.FAIL) + "Prob(conficker) = " + str(probs[0][9]) + bcolors.ENDC
		print "%-20s%-40s%-35s%-50s" % ("Iteration " + str(total),
										colored_str,
										failed_str,
										bcolors.OKBLUE + "Cumulative Success Rate: " + str(float(conficker)/total) + bcolors.ENDC)

	print bcolors.OKBLUE + "\n-----------------------------------------------------------------------------\n\n\n\n"
	print bad
	print "\n\n\n\n\n\n"
	print "Accuracy: %s/%s = %s" % (str(conficker), str(total), str(float(conficker)/total))
	print bcolors.ENDC

def test_binary(file_name):
	domain_list = init(file_name)
	mcnt = 0
	conc = []
	success = 0
	total = 0
	for domain in domain_list:
		newvec = [0]*len(cols1)
		for i in range(len(cols1)):
			if cols1[i] in domain:
				newvec[i] = 1
		# if len(domain) > 15:
			# newvec.append(1)
		# else:
			# newvec.append(0)

		probs = binary_model.predict_proba(np.array([newvec]), verbose=0)[0]
		
		final = ((bcolors.FAIL + "Malignant" + bcolors.ENDC) if probs[1] > probs[0] else (bcolors.OKGREEN + "Benign" + bcolors.ENDC))
		mcnt = (mcnt + 1 if probs[1] > probs[0] else mcnt)
		prob_formatted = "[" + bcolors.OKGREEN + str(probs[0]) + bcolors.ENDC + ", " + bcolors.FAIL + str(probs[1]) + bcolors.ENDC + "]; "
		
		if (probs[1] > probs[0]):
			success += 1
		total += 1

		print "%-15s %-45s%-30s %-35s%-50s" % (domain,
										 prob_formatted,
										 "Conclusion = " + final,
										 "Failed = (" + str(total - success) + ", " + str(float(total-success)/total) + ")",
										 bcolors.OKBLUE + "Cumulative Success Rate: " + str(float(success)/total) + bcolors.ENDC)
	
	print "\n"
	print "Accuracy: %s/%s = %s" % (str(mcnt), str(domain_list.__len__()), str(float(mcnt)/domain_list.__len__()))

def test_multi_RNN(file_name):

	domain_list = init(file_name)
	domain_list = [[valid_chars[y] for y in x] for x in domain_list]
	domain_list = sequence.pad_sequences(domain_list, maxlen=maxlen)
	fail = 0
	total = 0
	for domain in domain_list:
		probs = multi_RNN_model.predict_proba(np.array([domain]), verbose=0)[0]
		color = bcolors.OKGREEN if np.amax(probs) == probs[2] else bcolors.FAIL
		total += 1
		if np.amax(probs) != probs[2]:
			fail += 1
		np.set_printoptions(precision=4)
		np.set_printoptions(threshold=5)
		print ("%-20s" % (str(total-fail) + "/" + str(total))) + bcolors.HEADER + ("%-30s" % ("Accuracy: " + str(100.0*float(total-fail)/total) + "%")) + bcolors.ENDC + "Probability of conficker: " + color + str(probs[2]*100) + "%" + "\t" + str(probs) + bcolors.ENDC

	# X = [x[1] for x in indata]
	# labels = [x[0] for x in indata]

	# print "len(X): " + str(X.__len__())
	# print "len(labels): " + str(labels.__len__())

	# valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}
	# valid_labels = {x:idx for idx, x in enumerate(set(labels))}

	# max_features = len(valid_chars) + 1
	# maxlen = np.max([len(x) for x in X])

	# print "len(max_features): " + str(max_features)

	# # Convert characters to int and pad
	# X = [[valid_chars[y] for y in x] for x in X]
	# X = sequence.pad_sequences(X, maxlen=maxlen)


def main(model_type, file_name):
	if model_type == 'multi':
		test_multi(file_name)
	elif model_type == 'multi_RNN':
		test_multi_RNN(file_name)
	else:
		test_binary(file_name)

if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2])
