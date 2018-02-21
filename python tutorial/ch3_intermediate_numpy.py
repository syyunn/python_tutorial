# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:39:57 2017

@author: gihun
"""

import numpy as np
np.set_printoptions(linewidth=100, threshold=2000)

#%% auto expansion (broadcasting rule)
# https://docs.scipy.org/doc/numpy-1.10.0/user/basics.broadcasting.html

a = np.reshape(np.arange(0,12,1),[3,4])
b = np.ones([1,4])
c = a+b
print(c)

b = np.ones(4)
c = a+b
print(c)

b = 2*np.ones([3,1])
print(b)
print(a)
c = a+b
print(c)

#b= np.ones(2)
#c = a+b # err. cannot broadcast together with (3,4) (2,)

a = np.array([[1],[2],[3]])
print(a, a.shape, a.ndim)

b = np.array([[1,2,3,5]])
print(b, b.shape, b.ndim)

c = a+b
print(c, c.shape, c.ndim)

#%% all numpy object is pointer, call by reference

a = np.ones([5,5])
b = np.ones([5,5])
# a, b itself is pointer

def f(a, b):
    a = a+b # new pointer assigned
f(a,b)
print(a)
print(b)

def f(a,b):
    a[:,:]=a+b # at pointer's data value assigned
    
f(a,b)
print(a)
print(b)

#%% useful operation
a = np.reshape(np.arange(0,24,1),[2,3,4])

# np.sum( tf.reduce_sum)
print(np.sum(a))

b = np.sum(a, axis=0)
print(b, b.shape)

b = np.sum(a, axis=1)
print(b, b.shape)

b = np.sum(a, axis=2)
print(b, b.shape)

b = np.sum(a, axis=1, keepdims=True)
print(b, b.shape)

# np.mean (tf.reduce_mean)
print(np.mean(a))

b = np.mean(a, axis=0)
print(b, b.shape)

# np.prod (tf.reduce_prod)
b = np.prod(a, axis=0)
print(b, b.shape)

#%% misc ( max min argmax ...)
np.maximum(a, 10) # tf.maximum

print(np.max(a)) # tf.reduce_max

b = np.max(a, axis=0) # tf.reduce_max
print(b, b.shape)

b = np.max(a, axis=1)
print(b, b.shape) # tf.reduce_max

b = np.max(a, axis=1, keepdims=True) # tf.reduce_max
print(b, b.shape)

b = np.min(a, axis=0) # tf.reduce_min
print(b, b.shape)

b = np.argmax(a, axis=0) # tf.argmax
print(b, b.shape)

b = np.argmax(a, axis=2) # tf.argmax
print(b, b.shape)

b = np.argmax(a, axis=2) # tf.argmax, (use tf.argmax rather than tf.arg_max, tf.arg_max will be deprecated)
print(b, b.shape)

a = np.tile(a, [2,3]) # tf.tile
print(a)
a = a[:,0:3,0:4]
a = np.tile(a,[4])
print(a)
a.shape

# dimension expansion trick
b = np.array([1,2,3])
print(b, b.shape) # (3)

b = b[None,:,None] # (1,3,1)
print(b, b.shape)

b = np.tile(b, [3,1,3])
print(b, b.shape) # (3,3,3)



#%%
Quiz
Q1.(auto expansion) what is result of print()?
a = np.array([[0],[1],[2]])
b = np.array([[0,1,2]])
c = a+b
print(c, c.shape, c.ndim)








A1.
[[0 1 2]
 [1 2 3]
 [2 3 4]] (3, 3) 2

 
 
Q2. what is result of a?
def f(a,b):
    a = a+b
a = np.ones([1])
b = np.ones([1])
f(a,b)
print(a)
def f2(a,b):
    a[:] = a+b
f2(a,b)
print(a)








A2.
[1.]
[2.]






Q3.
a = np.reshape(np.arange(0,24,1),[2,3,4])
how to get sum when axes are 0, 1?
--> it must be shaped [1,1,4]
import numpy as np





A3.
np.sum(a, axis=(0,1),keepdims=True) # tuple





Q4.
a = np.ones([2,2])
b = np.zeros_like(a)
make below array with a, b
[[ 1.  0.  1.  0.]
 [ 1.  0.  1.  0.]
 [ 1.  0.  1.  0.]
 [ 1.  0.  1.  0.]]
a
b
c = np.tile(np.concatenate((a,b),1),(2,1))
print (np.transpose(np.reshape(c,[4,4])))

A4.

c = np.stack([a,b],axis=2)
c = np.reshape(c, [4,2])
c = np.tile(c, [1,2])
print(c)


Q5.
a = np.ones([2,2])
b = np.zeros_like(a)
make below array with a, b
[[ 0.  1.  0.  1.]
 [ 1.  0.  1.  0.]
 [ 0.  1.  0.  1.]
 [ 1.  0.  1.  0.]]

 
 
c = np.stack([b,a],axis=2)
d = np.stack([a,b],axis=2)
e = np.stack([c,d],axis=1)
np.reshape(e, [4,4])
 
 
 
 
 
 
 
 
 
 
 
 
 
 
A5.
c = np.stack([a,b],axis=2)
c = np.reshape(c, [2,4])
c = np.tile(c, [2,1])
print(c)
d = c[0,:].tolist()
d.reverse()
c[::2,:]=d
print(c)





Q6.
a = np.reshape(np.arange(4),[2,2])
b = np.zeros_like(a)
b = np.reshape(np.arange(4,8,1),[2,2])
make below array with a, b
[[0 1 0 0]
 [2 3 0 0]]
###############
[[ 0.  1.  0.  1.  0.  0.]
 [ 2.  3.  2.  3.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]]
 
c = np.concatenate([a,b],1)  
d = [a,a,b,b,b,b]

e = np.stack(d, axis=0) # 6*2*2
np.stack(d, axis=1) # 2*6*2
np.stack(d, axis=2) # 2*2*6

np.split(e, axis=1)

 
 
 

 
 
 
 

 
print(np.concatenate([a,b],axis=1))
############## 
a = a[None,:]
b = np.zeros([4,2,2])
c = np.concatenate([a,a,b],axis=0)
c = np.reshape(c,[2,3,2,2])
d = [np.squeeze(a) for a in np.split(c,2,axis=0)]
d = np.concatenate(d,axis=1)
e = [np.squeeze(a) for a in np.split(d,3,axis=0)]
f = np.concatenate(e, axis=1)
print(f)
