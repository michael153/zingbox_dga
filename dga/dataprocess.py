#!/usr/bin/env python
#
#  __  /_)             |                
#     /  | __ \   _` | __ \   _ \\ \  / 
#    /   | |   | (   | |   | (   |`  <  
#  ____|_|_|  _|\__, |_.__/ \___/ _/\_\ 
#               |___/                   
#
#  (C) Copyright Zingbox Ltd 2017
#  All Rights Reserved

import os
import random
import tldextract
import glob
import json
import ast
import pickle
import ipaddress
import tldextract
import numpy as np 
import pandas as pd
import cPickle as pickle
from datetime import datetime
from StringIO import StringIO
from urllib import urlopen
from zipfile import ZipFile
from itertools import chain
from collections import Counter
from collections import defaultdict
from dga_generators import banjori, corebot, cryptolocker, \
    dircrypt, kraken, lockyv2, pykspa, qakbot, ramdo, ramnit, simda

class Dataprocess():

    def __init__(self, num):
        self.data_dir = os.path.abspath('data')
        self.datafile1 = os.path.join(self.data_dir, 'dga_train'+'.pkl')
        self.datafile2 = os.path.join(self.data_dir, 'all_train'+'.pkl')
        self.num = num

    def get_alexa(self):
        filename='top-1m.csv'
        address = 'http://s3.amazonaws.com/alexa-static/top-1m.csv.zip'
        url = urlopen(address)
        zipfile = ZipFile(StringIO(url.read()))
        return [tldextract.extract(x.split(',')[1]).domain for x in \
                zipfile.read(filename).split()[:3*self.num]]

    def get_malicious(self):
        num_per_dga=self.num
        domains = []
        labels = []

        # We use some arbitrary seeds to create domains with banjori
        banjori_seeds = ['somestring', 'firetruck', 'bulldozer', 'airplane', 'racecar',
                         'apartment', 'laptop', 'laptopcomp', 'malwareisbad', 'crazytrain',
                         'thepolice', 'fivemonkeys', 'hockey', 'football', 'baseball',
                         'basketball', 'trackandfield', 'fieldhockey', 'softball', 'redferrari',
                         'blackcheverolet', 'yellowelcamino', 'blueporsche', 'redfordf150',
                         'purplebmw330i', 'subarulegacy', 'hondacivic', 'toyotaprius',
                         'sidewalk', 'pavement', 'stopsign', 'trafficlight', 'turnlane',
                         'passinglane', 'trafficjam', 'airport', 'runway', 'baggageclaim',
                         'passengerjet', 'delta1008', 'american765', 'united8765', 'southwest3456',
                         'albuquerque', 'sanfrancisco', 'sandiego', 'losangeles', 'newyork',
                         'atlanta', 'portland', 'seattle', 'washingtondc']

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
            labels += ['cryptolocker']*segs_size
        
       # domains += dircrypt.generate_domains(num_per_dga)
       # labels += ['dircrypt']*num_per_dga
        '''
        #generate kraken and divide between configs
        kraken_to_gen = max(1, num_per_dga/2)
        domains += kraken.generate_domains(kraken_to_gen, datetime(2016, 1, 1), 'a', 3)
        labels += ['kraken']*kraken_to_gen
        domains += kraken.generate_domains(kraken_to_gen, datetime(2016, 1, 1), 'b', 3)
        labels += ['kraken']*kraken_to_gen
        '''
        # generate locky and divide between configs
        
        locky_gen = max(1, num_per_dga/11)
        for i in range(1, 12):
            domains += lockyv2.generate_domains(locky_gen, config=i)
            labels += ['locky']*locky_gen
        '''
        #Generate pyskpa domains
        domains += pykspa.generate_domains(num_per_dga, datetime(2016, 1, 1))
        labels += ['pykspa']*num_per_dga
        
        # Generate qakbot
        
        domains += qakbot.generate_domains(num_per_dga, tlds=[])
        labels += ['qakbot']*num_per_dga
        
        # ramdo divided over different lengths
        ramdo_lengths = range(8, 32)
        segs_size = max(1, num_per_dga/len(ramdo_lengths))
        for rammdo_length in ramdo_lengths:
            domains += ramdo.generate_domains(segs_size,
                                              seed_num=random.randint(1, 1000000),
                                              length=rammdo_length)
            labels += ['ramdo']*segs_size
        '''
        # ramnit
        domains += ramnit.generate_domains(num_per_dga, 0x123abc12)
        labels += ['ramnit']*num_per_dga
        '''
        # simda
        simda_lengths = range(8, 32)
        segs_size = max(1, num_per_dga/len(simda_lengths))
        for simda_length in range(len(simda_lengths)):
            domains += simda.generate_domains(segs_size,
                                              length=simda_length,
                                              tld=None,
                                              base=random.randint(2, 2**32))
            labels += ['simda']*segs_size
        '''
        return domains, labels

    def get_external(self):

        domains = []
        labels = []
        external_path = os.path.abspath('dga_wordlists')
        
        conficker = []
        with open(os.path.join(external_path,'conficker.txt'), 'r') as f:
            for line in f:
                conficker.append(tldextract.extract(line).domain)
        domains += conficker[:self.num]
        labels += ['conficker']*self.num
        
        zeus = []
        with open(os.path.join(external_path,'zeus.txt'), 'r') as f:
            for line in f:
                zeus.append(tldextract.extract(line).domain)
        domains += zeus[:self.num]
        labels += ['zeus']*self.num
        
        tinba = []
        with open(os.path.join(external_path,'tinba.txt'), 'r') as f:
            for line in f:
                tinba.append(tldextract.extract(line).domain)
        domains += tinba[:self.num]
        labels += ['tinba']*self.num

        matsnu = []
        with open(os.path.join(external_path,'matsnu.txt'), 'r') as f:
            for line in f:
                matsnu.append(tldextract.extract(line).domain)
        domains += matsnu[:self.num]
        labels += ['matsnu']*self.num
        
        rovnix = []
        with open(os.path.join(external_path,'rovnix.txt'), 'r') as f:
            for line in f:
                rovnix.append(tldextract.extract(line).domain)
        domains += rovnix[:self.num]
        labels += ['rovnix']*self.num
        
        pushdo = []
        with open(os.path.join(external_path,'pushdo.txt'), 'r') as f:
            for line in f:
                pushdo.append(tldextract.extract(line).domain)
        domains += pushdo[:self.num]
        labels += ['pushdo']*self.num

        goz = []
        with open(os.path.join(external_path,'goz.txt'), 'r') as f:
            for line in f:
                goz.append(tldextract.extract(line).domain)
        domains += goz
        labels += ['goz']*len(goz)
        
        return domains, labels

    def gen_data(self, force=False):

        if force or (not os.path.isfile(self.datafile1)) or (not os.path.isfile(self.datafile2)):
            domains, labels = self.get_malicious()
            otherdomains, otherlabels = self.get_external()
            domains += otherdomains
            labels += otherlabels
            pickle.dump(zip(labels, domains), open(self.datafile1, 'w'))
            labels = ['malicious']*len(labels)
            # Get equal number of benign/malicious
            exdomains = self.get_alexa()
            domains += exdomains
            labels += ['benign']*len(exdomains)           
            pickle.dump(zip(labels, domains), open(self.datafile2, 'w'))

    def get_data(self, type, force=False):
        self.gen_data(force)
        if type == 'multi':
            return pickle.load(open(self.datafile1))
        else:
            return pickle.load(open(self.datafile2))
