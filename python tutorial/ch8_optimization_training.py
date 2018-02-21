# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:31:06 2017

@author: gihun
"""
import tensorflow as tf
#%% optimizer
learning_rate = 1e-4
tf.train.GradientDescentOptimizer(learning_rate)
tf.train.AdamOptimizer().minimize(loss, var_list=trainables)
tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0).minimize(loss, var_list="trainables")

'''
basically, An optimizer needs learning_rate, loss, and var_list 
'''
#%% loss
# cross entropy

tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)
'''
logits and labels must have the same shape [batch_size, num_classes] and the same dtype (either float16, float32, or float64).
ex ) if you want it using to solve mnist problem, you can design model's output shape - [batch, num_classes(0~9:one_hot encoding)]
and label shape - [batch, num_classes(0~9:one_hot encoding)]. 
'''
tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
'''
logits: Unscaled log probabilities of shape
[d_0, d_1, ..., d_{r-1}, num_classes] and dtype float32 or float64.
A common use case is to have logits of shape [batch_size, num_classes] and labels of shape [batch_size]. But higher dimensions are supported.
ex ) In the same above case, you can design model's output shape -[batch, num_classes(0~9:one_hot encoding)]
and label shape - [batch] each result has element(0~9)
'''

# l2 loss
distance = x - label
tf.reduce_mean(tf.reduce_sum(0.5*distance*distance, reduction_indices=[1,2,3]))
'''
it may be acustomed to you....
label would be img
output would be model's output img
then l2 loss would be above code.
and you can also apply it at various area such as vgg loss.
'''

#%%
tf.train.Saver()
tf.train.Saver().save
tf.train.Saver().restore


#%%
import tensorflow as tf
import numpy as np
from ch4_imshow_opencv import *

#%%
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#%%
tf.reset_default_graph()
labels=tf.placeholder(dtype=tf.float32,shape=[None,10])
x = tf.placeholder(dtype=tf.float32,shape=[None,28,28,1])
with tf.variable_scope("conv1"):
    W1 = tf.get_variable(name="w",initializer=tf.truncated_normal(shape=[3,3,1,64],mean=0,stddev=0.01,dtype=tf.float32))
    b1 = tf.get_variable(name="b",initializer=tf.zeros([64],dtype=tf.float32))
    conv1=tf.nn.conv2d(x,W1,strides=[1,1,1,1],padding='SAME')+b1
    relu1=tf.nn.relu(conv1, name="relu")
    pool1 = tf.nn.max_pool(relu1,[1,2,2,1],strides=[1,2,2,1],padding='SAME', name="pool")
with tf.variable_scope("conv2"):
    W2 = tf.get_variable(name="w",initializer=tf.truncated_normal(shape=[3,3,64,64],mean=0,stddev=0.01,dtype=tf.float32))
    b2 = tf.get_variable(name="b",initializer=tf.zeros([64],dtype=tf.float32))
    conv2=tf.nn.conv2d(pool1,W2,strides=[1,1,1,1],padding='SAME')+b2
    relu2=tf.nn.relu(conv2, name="relu")
    pool2 = tf.nn.max_pool(relu2,[1,2,2,1],strides=[1,2,2,1],padding='SAME', name="pool")
with tf.variable_scope("conv3"):    
    W3 = tf.get_variable(name="w", initializer=tf.truncated_normal(shape=[3,3,64,64],mean=0,stddev=0.01,dtype=tf.float32))
    b3 = tf.get_variable(name="b", initializer=tf.zeros([64],dtype=tf.float32))
    conv3=tf.nn.conv2d(pool2,W3,strides=[1,1,1,1],padding='SAME')+b3
    flatten=tf.reshape(conv3,(-1,7*7*64), name="flatten")
with tf.variable_scope("full4"):
    W4 = tf.get_variable(name="w", initializer=tf.truncated_normal(shape=[49*64,10],mean=0,stddev=0.01,dtype=tf.float32))
    b4 = tf.get_variable(name="b", initializer=tf.zeros([10],dtype=tf.float32))
    logits= tf.add(tf.matmul(flatten,W4),b4, name="logits")
'''
'''
trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print(trainables)

loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)
loss = tf.reduce_mean(loss)
learning_rate = 1e-3
trainop = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=trainables)


sess = tf.Session()
#%%
sess.run(tf.global_variables_initializer())

#%%
import time
a = time.time()
for i in range(21):
    batchx,batchlabel = mnist.train.next_batch(128)
    batchx = np.reshape(batchx,[128,28,28,1])
    feed_dict = {x:batchx,labels:batchlabel}
    _,myloss,logits_eval=sess.run([trainop,loss,logits],feed_dict = feed_dict)
    logits_eval=np.argmax(logits_eval,axis=1)
    batchlabel_eval = np.argmax(batchlabel,axis=1)
    accuracy= np.sum((batchlabel_eval==logits_eval))/128
    print("[train]",i,myloss,accuracy)
	
    if i !=0 and i%100 ==0:
        batchx,batchlabel = mnist.test.next_batch(20)
        batchx = np.reshape(batchx,[20,28,28,1])
        feed_dict = {x:batchx,labels:batchlabel}
        myloss,logits_eval = sess.run([loss,logits],feed_dict=feed_dict)
        logits_eval = np.argmax(logits_eval,axis=1)
        batchlabel_eval = np.argmax(batchlabel,axis=1)
        accuracy = np.sum((batchlabel_eval==logits_eval))/20
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")    
        print("[test]",i,myloss, accuracy)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

b = time.time()
c = b-a
print(c)
#%%
batchx,batchlabel = mnist.test.next_batch(50)
batchx = np.reshape(batchx,[50,28,28,1])
feed_dict = {x:batchx,labels:batchlabel}

myloss,logits_eval = sess.run([loss,logits],feed_dict=feed_dict)
logits_eval = np.argmax(logits_eval,axis=1)
batchlabel_eval = np.argmax(batchlabel,axis=1)
imshow4d(batchx,figsize=(10,5),nx=5)
acc = np.sum((batchlabel_eval==logits_eval))/50
print(np.reshape(logits_eval,[-1,5]))
print(np.reshape(batchlabel_eval,[-1,5]))
print(logits_eval-batchlabel_eval)
print("[test]",myloss, acc)

#%% save
variables = tf.get_collection("variables")
save_path = "save\\ch7_mnist"
import os
#os.mkdir(save_path)
saver=tf.train.Saver(variables,max_to_keep=10)
saver.save(sess, os.path.join(save_path, "model"),i)

#%% restore
saver = tf.train.Saver(variables,max_to_keep=10)
saver.restore(sess,os.path.join(save_path, "model-1000"))
