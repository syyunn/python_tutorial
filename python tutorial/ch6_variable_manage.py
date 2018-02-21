# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:30:32 2017
Variable 

@author: gihun
"""
import tensorflow as tf
import numpy as np
#tf.variable_scope
#tf.get_collection
#tf.add_to_collection
#%%
class Eval():
    def __init__(self):
        self.sess = tf.Session()
    def global_var_init(self):
        self.sess.run(tf.global_variables_initializer())
    def __call__(self, var, feed_dict=None):
        return self.sess.run(var, feed_dict)
                 

#%% tf.get_variable & reuse issue
'''
1. First is that tf.Variable will always create a new variable, whether tf.get_variable gets from the graph 
an existing variable with those parameters, and if it does not exists, it creates a new one.
2. tf.Variable requires that an initial value be specified.
'''
tf.reset_default_graph()
x = tf.Variable(tf.zeros([4,4]), name="x",trainable=False)
x = tf.Variable(tf.ones([4,4]), name="x")
y = tf.get_variable(name="x",initializer=np.ones([4,4],dtype=np.float32))
y = tf.get_variable(name="x",initializer=np.zeros([4,4],dtype=np.float32)) # error reuse False
tf.get_variable_scope().reuse_variables() #Not recommend, if you use it, you cannot recover the state of it. 
tf.get_variable_scope().reuse
z = tf.get_variable(name="x")
with tf.variable_scope(tf.get_variable_scope(), reuse=False):
    print(tf.get_variable_scope().reuse)
    w = tf.get_variable(name="w", initializer=np.ones([4,4],dtype=np.float32)) # error reuse True --> Cannot generate
e = Eval()
e.global_var_init()
e([x,y,z])
e([x])
#%% tf.variable_scope and reuse issue

tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
tf.get_collection("variables")
tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
tf.reset_default_graph()

with tf.variable_scope("one",reuse=False):
    a = tf.get_variable("v", [1], initializer=tf.random_normal_initializer) #v.name == "one/v:0"
    with tf.variable_scope(tf.get_variable_scope()):
        a2 = tf.get_variable("w",[1], initializer=tf.random_normal_initializer) 
    b = tf.get_variable("v", [1]) #w.name == "one/v:0" #ValueError: Variable one/v already exists
with tf.variable_scope("one", reuse = True):
    c = tf.get_variable("v")

with tf.variable_scope("two"):
    d = tf.get_variable("v", [1], initializer=tf.constant_initializer([0])) #z.name == "two/v:0"
    e2 = tf.Variable([1], name="v")

with tf.variable_scope("one"):
    with tf.variable_scope("two"):
        f = tf.get_variable("v", [1], initializer=tf.contrib.layers.xavier_initializer_conv2d())
assert(a == c)  #Assertion is true, they refer to the same object.
assert(a == d)  #AssertionError: they are different objects
assert(d == e)  #AssertionError: they are different objects
eval = Eval()
eval.global_var_init()
eval([a,a2,c,d,e2,f])

tf.get_collection("variables")
tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="one")
tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="two")


#%% getting tensors by name
g = tf.get_default_graph()
copy_a =g.get_tensor_by_name("one"+"/"+"v"+":0")
copy_a2 = g.get_tensor_by_name("one"+"/"+"v"+":0")
copy_c = copy_a
copy_d = g.get_tensor_by_name("two"+"/"+"v"+":0")
copy_e2 = g.get_tensor_by_name("two/"+"v:0")
copy_f = g.get_tensor_by_name("one/two"+"/"+"v"+":0")
eval([copy_a,copy_a2,copy_c,copy_d,copy_e2,copy_f])


#%% * get_collection & add_to_collection * management of variable list
tf.reset_default_graph()
with tf.variable_scope("conv1"):
    x = tf.placeholder(name="x",dtype=tf.float32,shape=[None,None,None,1])
    w = tf.get_variable(name="w", shape=[3,3,1,5], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b = tf.get_variable(name="b",initializer=tf.zeros(w.get_shape()[3], dtype=tf.float32))
    conv = tf.add(tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME"), b, name="conv")
    relu = tf.nn.relu(conv, name="relu")
    flatten = tf.reshape(relu, [-1,4*4*5],name="flatten") # we already know input size will be [b,4,4,3]
with tf.variable_scope("full1"):
    x_full = flatten
    w = tf.get_variable(name="w",shape=[80,3], dtype=tf.float32, initializer=tf.random_normal_initializer)
    b = tf.get_variable(name="b",shape=w.get_shape()[1], dtype=tf.float32)    
    full = tf.add(tf.matmul(x_full,w),b, name="full")

    
tf.get_collection("variables")
tf.get_collection("variables", scope="conv1")
tf.get_collection("variables", scope="full1")

import re
glo_vars = tf.get_collection("variables")
w_vars = [v for v in glo_vars if re.match("^.*"+"/.*w:0$", v.name)]
b_vars = [v for v in glo_vars if re.match("^.*"+"/.*b:0$", v.name)]

tf.add_to_collection("weights",w_vars)
tf.add_to_collection("bias",b_vars)

ws = tf.get_collection("weights")
bs = tf.get_collection("bias")


x_ = np.identity(4)
x_ = np.stack([x_,x_+1], axis=0)[:,:,:,None]
feed_dict = {x:x_}
e = Eval()
e.global_var_init()
e(full, feed_dict)

g = tf.get_default_graph()
copy_full = g.get_tensor_by_name("full1/full"+":0")
e(copy_full,feed_dict)
copy_relu= g.get_tensor_by_name("conv1/relu"+":0")
e(copy_relu,feed_dict).shape


#%% save and restore
tf.reset_default_graph()
with tf.variable_scope("conv1"):
    x = tf.placeholder(name="x",dtype=tf.float32,shape=[None,None,None,1])
    w = tf.get_variable(name="w", shape=[3,3,1,5], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b = tf.get_variable(name="b",initializer=tf.zeros(w.get_shape()[3], dtype=tf.float32))
    conv = tf.add(tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME"), b, name="conv")
    relu = tf.nn.relu(conv, name="relu")
    flatten = tf.reshape(relu, [-1,4*4*5],name="flatten") # we already know input size will be [b,4,4,3]
with tf.variable_scope("full1"):
    x_full = flatten
    w = tf.get_variable(name="w",shape=[80,3], dtype=tf.float32, initializer=tf.random_normal_initializer)
    b = tf.get_variable(name="b",shape=w.get_shape()[1], dtype=tf.float32)    
    full = tf.add(tf.matmul(x_full,w),b, name="full")
    
variables = tf.get_collection("variables")
saver = tf.train.Saver(variables, max_to_keep=10)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
import os
os.mkdir("save\\test")
save_path = os.path.join(os.getcwd(), "save\\test")
saver.save(sess,os.path.join(save_path,"model"),0)

#%%
tf.reset_default_graph()
with tf.variable_scope("conv1"):
    x = tf.placeholder(name="x",dtype=tf.float32,shape=[None,None,None,1])
    w = tf.get_variable(name="w", shape=[3,3,1,5], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b = tf.get_variable(name="b",initializer=tf.zeros(w.get_shape()[3], dtype=tf.float32))
    conv = tf.add(tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME"), b, name="conv")
    relu = tf.nn.relu(conv, name="relu")
    flatten = tf.reshape(relu, [-1,4*4*5],name="flatten") # we already know input size will be [b,4,4,3]
with tf.variable_scope("full1"):
    x_full = flatten
    w = tf.get_variable(name="w",shape=[80,3], dtype=tf.float32, initializer=tf.random_normal_initializer)
    b = tf.get_variable(name="b",shape=w.get_shape()[1], dtype=tf.float32)    
    full = tf.add(tf.matmul(x_full,w),b, name="full")

variables = tf.get_collection("variables")
saver = tf.train.Saver(variables, max_to_keep=10)
e = Eval()

save_path = os.path.join(os.getcwd(), "save\\test")
saver.restore(e.sess,os.path.join(save_path,"model-0"))

x_ = np.identity(4)
x_ = np.stack([x_,x_+1], axis=0)[:,:,:,None]
feed_dict = {x:x_}

e(full,feed_dict)