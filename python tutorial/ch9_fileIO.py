# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 22:19:01 2017

@author: gihun
"""
#%% open with "w" and write
f2 = open("STUDY\\inputs\\newfile.txt",'w')
for i in range(1,11):
    data = "%d번째 줄입니다.\n" %i
    f2.write(data)
f2.close()

#%% readline() open with "r"
f3 = open("STUDY\\inputs\\새파일.txt",'r')
while True:
    line = f3.readline()
    if not line: break
    print(line)
f3.close
#%% readlines()
f4 = open("STUDY\\inputs\\새파일.txt",'r')
lines = f4.readlines()
for line in lines:
    print(line)
f4.close()
lines
#%% read()
f = open("STUDY\\inputs\\새파일.txt",'r')
data = f.read()
print(data)
f.close()    

#%% add contents
f = open("STUDY\\inputs\\새파일.txt",'a')
for i in range(15,20):
    data = "%d번째 줄입니다.\n" %i
    f.write(data)
f.close()

#%% 
f = open("STUDY\\inputs\\강아지.txt",'w')
lst = []
f2 = open("STUDY\\inputs\\새파일.txt",'r')
lines = f2.readlines()
lst.append("제목:강아지\n")
lst.append("작곡:검정치마\n")
for line in lines:
    lst.append(line)
f.writelines(lst)
f.close()   

#%%
f = open("STUDY\\inputs\\강아지.txt",'r+') 
lines = f.readlines()
lines.append('--중략--')
f.writelines(lines)
f.close()
#%%
f = open("STUDY\\inputs\\foo.txt",'w')
f.write("Life is short, you need python")
f.close()

#%%
with open("STUDY\\inputs\\foo.txt",'w') as f:
    f.write("Life is short, you need python3")

#%%
with open("STUDY\\inputs\\data.txt","w") as f:
    for i in range(10):
        data = '%d' %i
        f.write(data)
f.close()
#%%

import utils
from utils.core import *
helper_info()
