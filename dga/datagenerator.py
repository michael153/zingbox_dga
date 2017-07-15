import os
import json
import numpy as np
import pandas as pd
from dataprocess import Dataprocess
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

import numpy as np 
import pandas as pd

from datetime import datetime
from StringIO import StringIO
from urllib import urlopen
from zipfile import ZipFile
from itertools import chain
from collections import Counter
from collections import defaultdict
from string import ascii_lowercase, digits
from random import choice


from dga_generators import banjori, corebot, cryptolocker, \
    dircrypt, kraken, lockyv2, pykspa, qakbot, ramdo, ramnit, simda

import cPickle as pickle
import os
import time
import random
import datetime
import tldextract
import glob
import json
import ast
import pickle
import ipaddress
import tldextract


#!/usr/bin/env python
#
#  __  /_)             |                
#     /  | __ \   _` | __ \   _ \\ \  / 
#    /   | |   | (   | |   | (   |`  <  
#  ____|_|_|  _|\__, |_.__/ \___/ _/\_\ 
#               |___/                   
#
#  (C) Copyright Zingbox Ltd 2016
#  All Rights Reserved

class Datagenerator():

    #logger = logging.getLogger(__name__)

    def __init__(self, min_num, max_num):
        #self.env = env
        #self.tenantid = tenantid
        self.data_dir = os.path.abspath('data')
        self.test_data = os.path.join(self.data_dir, 'all_test'+'.pkl')
        self.max_num = max_num
        self.min_num = min_num
        self.num = max_num - min_num

    def get_alexa(self):
    	filename='top-1m.csv'
        address = 'http://s3.amazonaws.com/alexa-static/top-1m.csv.zip'
        url = urlopen(address)
        zipfile = ZipFile(StringIO(url.read()))
        return [tldextract.extract(x.split(',')[1]).domain for x in \
                zipfile.read(filename).split()[:500]]

    def get_malicious(self):
        cur_date = datetime.datetime(datetime.datetime.today().year, datetime.datetime.today().month, datetime.datetime.today().day)  

        num_per_dga = max(1, self.max_num/30)
        max_num = self.max_num
        min_num = self.min_num
        """Generates num_per_dga of each DGA"""
        
        domains = []
        labels = []

        for delta in range(0, 30):
            print bcolors.WARNING + "Generating for timedelta(days=" + str(i) + ")" + bcolors.ENDC
            d = cur_date + datetime.timedelta(days=delta)

            banjori_seeds = []
            for k in range(0, 52):
                random.seed(time.time())
                total_len = random.randint(6, 15)
                s = ''.join(choice(ascii_lowercase + digits) for i in range(total_len))
                banjori_seeds.append(s)

            segs_size = max(1, num_per_dga/len(banjori_seeds))
            for banjori_seed in banjori_seeds:
                domains += banjori.generate_domains(segs_size, banjori_seed)
                labels += ['banjori']*segs_size
            
            domains += corebot.generate_domains(num_per_dga)
            labels += ['corebot']*num_per_dga
            
            #Create different length domains using cryptolocker
            crypto_lengths = range(8, 32)
            segs_size = max(1, num_per_dga/len(crypto_lengths))
            for crypto_length in crypto_lengths:
                domains += cryptolocker.generate_domains(segs_size,
                                                         seed_num=random.randint(1, 1000000),
                                                         length=crypto_length)
            
            locky_gen = max(1, num_per_dga/11)
            for i in range(1, 12):
                domains += lockyv2.generate_domains(locky_gen, config=i)
                labels += ['locky']*locky_gen

            domains += ramnit.generate_domains(num_per_dga, 0x123abc12)
            labels += ['ramnit']*num_per_dga

            domains += GameoverZeus.generate_domains(num_per_dga, d)
            labels += ['zeus']*num_per_dga

            domains += tinba.generate_domains(num_per_dga, d)
            labels += ['tinba']*num_per_dga

            domains += matsnu.generate_domains(num_per_dga, d)
            labels += ['matsnu']*num_per_dga

            domains += rovnix.generate_domains(num_per_dga, d)
            labels += ['rovnix']*num_per_dga

            domains += pushdo.generate_domains(num_per_dga, d)
            labels += ['pushdo']*num_per_dga

        if num_per_dga*30 != max_num:
            domains += pushdo.generate_domains(max_num - num_per_dga*30, cur_date)
            labels += ['pushdo']*(max_num - num_per_dga*30)

        final_domains = []
        final_labels = []
        
        for i in range(9):
        	final_domains.extend(domains[(max_num*i+min_num):(max_num*(i+1))])
        	final_labels.extend(labels[(max_num*i+min_num):(max_num*(i+1))])
        return final_domains, final_labels
    '''
    def get_internal(self):
        fex_table = pd.DataFrame().from_csv(os.path.join(self.data_dir,'fex_table_'+self.tenantid+'.csv'))
        internal_domain = []
        for feature in ['f21', 'f22']:
            mylist = fex_table[feature + '_list'].tolist()   
            mylist = [ast.literal_eval(x) for x in mylist if str(x) != 'nan']
            clist = [tldextract.extract(str(item)).domain for sublist in mylist for item in sublist if not is_ip(item)]
            counter = Counter(clist)
            internal_domain.extend([k for k in counter if counter[k] > 5])
      return list(set(internal_domain))
    '''

    def get_external(self):
        domains = []
        labels = []
        external_path = os.path.abspath('dga_wordlists')
        
        conficker = []
        with open(os.path.join(external_path,'conficker.txt'), 'r') as f:
            for line in f:
                conficker.append(tldextract.extract(line).domain)
        print bcolors.WARNING + "Getting random " + str(self.num) + " conficker data values..." + bcolors.ENDC
        random.shuffle(conficker)
        domains += conficker[:self.num]
        labels += ['conficker']*self.num
        
        return domains, labels

    def gen_data(self, force=False):
        if force or (not os.path.isfile(self.datafile1)) or (not os.path.isfile(self.datafile2)):
            domains, labels = self.get_malicious()
            otherdomains, otherlabels = self.get_external()
            domains += otherdomains
            labels += otherlabels
            # Get equal number of benign/malicious
            good_domains = self.get_alexa()
            domains += good_domains
            labels += ['Benign']*len(good_domains)        
            '''
            if self.internal:
                indomains = self.get_internal()
                indomains = [x for x in indomains if x not in exdomains]
                domains += indomains
                labels += ['benign']*len(indomains)
            '''
            pickle.dump(zip(labels, domains), open(self.test_data, 'w'))

    def get_data(self, force=False):
        self.gen_data(force)
        return pickle.load(open(self.test_data))

