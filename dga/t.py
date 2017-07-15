import os
import time
import random
import tldextract
import glob
import json
import ast
import pickle
import ipaddress
import tldextract
import datetime
import numpy as np 
import pandas as pd
import cPickle as pickle
import datetime
from string import ascii_lowercase, ascii_uppercase, digits
from StringIO import StringIO
from random import choice
from urllib import urlopen
from zipfile import ZipFile
from itertools import chain
from collections import Counter
from collections import defaultdict
from dga_generators import banjori, corebot, cryptolocker, \
    dircrypt, kraken, lockyv2, pykspa, qakbot, ramdo, ramnit, simda, GameoverZeus, \
    tinba, matsnu, rovnix, pushdo

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_malicious(val):
    cur_date = datetime.datetime(datetime.datetime.today().year, datetime.datetime.today().month, datetime.datetime.today().day)  
    num_per_dga = max(1, val/30)
    domains = []
    labels = []

    for delta in range(0, 30):
        print bcolors.WARNING + "Generating for timedelta(days=" + str(delta) + ")" + bcolors.ENDC
        d = cur_date + datetime.timedelta(days=delta)

        banjori_seeds = []
        for k in range(0, 3):
            random.seed(time.time())
            total_len = random.randint(6, 15)
            s = ''.join(choice(ascii_lowercase + digits) for i in range(total_len))
            banjori_seeds.append(s)

        segs_size = max(1, num_per_dga/len(banjori_seeds))
        for banjori_seed in banjori_seeds:
            domains += banjori.generate_domains(segs_size, banjori_seed)
            labels += ['banjori']*segs_size
        
        random.seed(time.time())
        domains += corebot.generate_domains(num_per_dga, seed=''.join(choice(ascii_uppercase[:6] + digits) for i in range(9)), d=d)
        labels += ['corebot']*num_per_dga
        
        #Create different length domains using cryptolocker
        crypto_lengths = range(8, 32)
        segs_size = max(1, num_per_dga/len(crypto_lengths))
        for crypto_length in crypto_lengths:
            domains += cryptolocker.generate_domains(segs_size,
                                                     seed_num=random.randint(1, 1000000),
                                                     length=crypto_length)
        
        random.seed(time.time())
        domains += lockyv2.generate_domains(num_per_dga, d=d, config=random.randint(1, 11))
        labels += ['locky']*num_per_dga

        random.seed(time.time())
        domains += ramnit.generate_domains(num_per_dga, int(''.join(choice(ascii_uppercase[:6] + digits) for i in range(8)), 16))
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

    return domains, labels

def get_external(val):
    domains = []
    labels = []
    external_path = os.path.abspath('dga_wordlists')
    
    conficker = []
    with open(os.path.join(external_path,'conficker.txt'), 'r') as f:
        for line in f:
            conficker.append(tldextract.extract(line).domain)
    print bcolors.WARNING + "Getting random " + str(val) + " conficker data values..." + bcolors.ENDC
    random.shuffle(conficker)
    domains += conficker[:val]
    labels += ['conficker']*val
    
    return domains, labels