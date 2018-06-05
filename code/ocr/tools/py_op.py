# -*- coding: utf-8 -*-
"""
此文件用于常用python函数的使用
"""
import os
import json
import traceback
from collections import OrderedDict 
import random
from fuzzywuzzy import fuzz

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

################################################################################
### pre define variables
#:: enumerate
#:: raw_input
#:: listdir
#:: sorted
### pre define function
def mywritejson(save_path,content):
    content = json.dumps(content,indent=4,ensure_ascii=False)
    with open(save_path,'w') as f:
        f.write(content)

def myreadjson(load_path):
    with open(load_path,'r') as f:
        return json.loads(f.read())

def mywritefile(save_path,content):
    with open(save_path,'w') as f:
        f.write(content)

def myreadfile(load_path):
    with open(load_path,'r') as f:
        return f.read()

def myprint(content):
    print json.dumps(content,indent=4,ensure_ascii=False)

def rm(fi):
    os.system('rm ' + fi)

def mystrip(s):
    return ''.join(s.split())

def mysorteddict(d,key = lambda s:s, reverse=False):
    dordered = OrderedDict()
    for k in sorted(d.keys(),key = key,reverse=reverse):
        dordered[k] = d[k]
    return dordered

def mysorteddictfile(src,obj):
    mywritejson(obj,mysorteddict(myreadjson(src)))

def myfuzzymatch(srcs,objs,grade=80):
    matchDict = OrderedDict()
    for src in srcs:
        for obj in objs:
            value = fuzz.partial_ratio(src,obj)
            if value > grade:
                try:
                    matchDict[src].append(obj)
                except:
                    matchDict[src] = [obj]
    return matchDict

def mydumps(x):
    return json.dumps(content,indent=4,ensure_ascii=False)

def get_random_list(l,num=-1,isunique=0):
    if isunique:
        l = set(l)
    if num < 0:
        num = len(l)
    if isunique and num > len(l):
        return 
    lnew = []
    l = list(l)
    while(num>len(lnew)):
        x = l[int(random.random()*len(l))]
        if isunique and x in lnew:
            continue
        lnew.append(x)
    return lnew

def fuzz_list(node1_list,node2_list,score_baseline=66,proposal_num=10,string_map=None):
    node_dict = { }
    for i,node1 in enumerate(node1_list):
        match_score_dict = { }
        for node2 in node2_list:
            if node1 != node2:
                if string_map is not None:
                    n1 = string_map(node1)
                    n2 = string_map(node2)
                    score = fuzz.partial_ratio(n1,n2)
                    if n1 == n2:
                        node2_list.remove(node2)
                else:
                    score = fuzz.partial_ratio(node1,node2)
                if score > score_baseline:
                    match_score_dict[node2] = score
            else:
                node2_list.remove(node2)
        node2_sort = sorted(match_score_dict.keys(), key=lambda k:match_score_dict[k],reverse=True)
        node_dict[node1] = [[n,match_score_dict[n]] for n in node2_sort[:proposal_num]]
        print i,len(node1_list)
    return node_dict, node2_list

def swap(a,b):
    return b, a

def mkdir(d):
    path = d.split('/')
    for i in range(len(path)):
        d = '/'.join(path[:i+1])
        if not os.path.exists(d):
            os.mkdir(d)

