# https://raw.githubusercontent.com/andrewaeva/DGA/master/dga_algorithms/Tinba.py
import os
from random import choice
import random
import datetime
from string import ascii_lowercase
import time


def tinbaDGA(idomain, seed, cnt):
    suffix = ".com"
    domains = []

    count = cnt
    eax = 0
    edx = 0
    for i in range(count):
        buf = ''
        esi = seed
        ecx = 0x10
        eax = 0
        edx = 0
        for s in range(len(seed)):
            eax = ord(seed[s])
            edx += eax
        edi = idomain
        ecx = 0x0C
        d = 0
        while (ecx > 0):
            al = eax & 0xFF
            dl = edx & 0xFF
            al = al + ord(idomain[d])
            al = al ^ dl
            al += ord(idomain[d + 1])
            al = al & 0xFF
            eax = (eax & 0xFFFFFF00) + al
            edx = (edx & 0xFFFFFF00) + dl
            if al > 0x61:
                if al < 0x7A:
                    eax = (eax & 0xFFFFFF00) + al
                    buf += chr(al)
                    d += 1
                    ecx -= 1
                    continue
            dl += 1
            dl = dl & 0xFF
            edx = (edx & 0xFFFFFF00) + dl

        domain = buf + suffix
        domains.append(domain)
        idomain = domain
    return domains


def generate_domains(num_per_dga, date=None):
    if not date:
        date = datetime.datetime.now()
    harddomain = "ssrgwnrmgba.com"
    random.seed(date)
    seed = ''.join(choice(ascii_lowercase) for i in range(12))
    domains = tinbaDGA(harddomain, seed, num_per_dga)
    index = 0
    ret = []
    for domain in domains:
        index += 1
        ret.append(domain)
    # print index
    return ret


# generate_domains(10)