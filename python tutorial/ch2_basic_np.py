# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:17:58 2017
Chapter 2
basic of numpy 
@author: gihun
"""
import numpy as np

np.set_printoptions(linewidth=100, threshold=2000)
#%% how to make nd array
b = [10,20,30]
b
a = np.array([10,20,30,40.0])
print(a)

b = np.zeros([5,5], dtype=np.float32) # tf.zeros
print(b)

c = np.zeros_like(a) # tf.zeros_like # shape, dtype
print(c)

b = np.ones([5,5], dtype=np.float32) # tf.ones
print(b)
c = np.ones_like(a) # tf.ones_like #

d = np.identity(3, dtype=np.float32)

print(d)   ### 항상 dtype 은 flaot32가 좋다. 64하면 안됨

a = np.arange(10, 0, -1)
print(a)

#%% basic operator

a = np.ones([5,5], dtype=np.float32)
b = np.ones([5,5], dtype=np.float32)

c = a+b # matrix add
print(c)

c = a*3
print(c)  # scalar mul

c = a*b # elementwise multiply
print(c)

c = a / (2*b) # elementwise div
print(c)

c = a@b # matrix mul
print(c)

#%% what is shape, rank

a = np.ones([2,3,4],dtype=np.float32)
print(a)
print(a.shape)
print(a.ndim)

a = np.ones([1,5],dtype=np.float32)
print(a)
print(a.shape)
print(a.ndim)
a[0,:]
a = np.ones([5],dtype=np.float32)
print(a)
print(a.shape)
print(a.ndim)

a = np.ones([],dtype=np.float32)
print(a)
print(a.shape)
print(a.ndim)

#%% how to reshape

a = np.arange(0,24,1)
b = np.reshape(a, [3,8]) # tf.reshape
print(b)
print(b.shape)
print(b.ndim)

b = np.reshape(a, [2,3,4]) # tf.reshape
print(b)
print(b.shape)
print(b.ndim)

#%% [:,:,::] 설명 및 인자 / [bool array]
a = np.reshape(np.arange(0,16,1),[4,4])
print(a)
print(a[1,2])

b = a[0:4:2, 0:4:2]
print(b)
print(b.shape)
print(b.ndim)

b = a[0::2, 0::2]
print(b)
print(b.shape)
print(b.ndim)

b = a[:4:2,:4:2]
print(b)
print(b.shape)
print(b.ndim)

b = a[1::2,1::2]
print(b)
print(b.shape)
print(b.ndim)
a.shape
b= a[:,[0,1]]
print(b)
print(b.shape)
print(b.ndim)

print(a>5)
print(a[a>5])
a[a>5]=0
print(a)
#%% how to concatenate

a = np.reshape(np.arange(0,12,1),[3,4])
b = np.reshape(np.arange(0,12,1),[3,4])
print(a)
c = np.concatenate([a,b],axis=0) # tf.concat
print(c)
print(c.shape)
print(c.ndim)
c = np.concatenate([a,b],axis=1) # tf.concat
print(c)
print(c.shape)
print(c.ndim)


#%% how to stack
a = np.reshape(np.arange(0,12,1),[3,4])
b = np.reshape(np.arange(0,12,1),[3,4])
c = np.stack([a,b],axis=0)
print(a)
print(c)
print(c.shape)
print(c.ndim)

c = np.stack([a,b],axis=1)
print(c)
print(c.shape)
print(c.ndim)

c = np.stack([a,b],axis=2)
print(c)
print(c.shape)
print(c.ndim)



#%% split
a = np.reshape(np.arange(0,18,1),[3,6])

print(a)
b = np.split(a, 3, axis=1) # tf.split, + tf.unstack
b
print(len(b))
for e in b:
    print(e, e.shape, e.ndim)

b = np.split(a, 3, axis=0)
print(len(b))
for e in b:
    print(e, e.shape, e.ndim)
    

#%% squeeze
a = np.reshape(np.arange(0,12,1),[1,3,4,1])
print(a, a.shape, a.ndim)
b = np.squeeze(a)
print(b)
print(b.shape)
print(b.ndim)

b = np.squeeze(a, axis=3)
print(b)
print(b.shape)
print(b.ndim)


#%% review Quiz
import numpy as np
Q1. 
A = np.array([1,2,3,4,5.0])
A.dtype??







A1. np.float64

Q2. what data type is useful on tensorflow environment?










A2. float32

Q3.make below matrix. what is the best efficient code?
[[0 1 2]
[3 4 0]
[0 0 0]]










a
A3.
a = np.reshape(np.arange(9,dtype=np.float32),[3,3])
a[a>4]=0
a


Q4.
a = np.reshape(np.arange(0,12,1),[3,4])
b = np.reshape(np.arange(0,12,1),[3,4])
c = np.stack([a,b],axis=2)


what is output of c.shape?

c.shape




A4. [3,2,4]

Q5.
a = np.reshape(np.arange(0,12,1),[3,4])
b = np.reshape(np.arange(0,12,1),[3,4])
c = np.stack([a,b],axis=0)

what is output of c.shape?

c.shape
[b,h,w,c]




A5. error : axis 2 is out of bounds for array of dimension 2


