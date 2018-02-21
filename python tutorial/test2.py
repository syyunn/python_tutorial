# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 17:39:40 2017

@author: gihun
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd

# three layers neural networks. Activation function is Sigmoid (logistic)
tf.reset_default_graph()
xyTraining = np.loadtxt('data\\ikheeTrainingNorm.txt', unpack=True).astype(np.float32)
xyTesting = np.loadtxt('data\\ikheeTestingNorm.txt', unpack=True).astype(np.float32)

x_data_training = np.transpose(xyTraining[0:-1])
y_data_training = np.reshape(xyTraining[-1], (len(x_data_training), 1))

x_data_testing = np.transpose(xyTesting[0:-1])
y_data_testing = np.reshape(xyTesting[-1], (len(x_data_testing), 1))

X = tf.placeholder(tf.float32, name='X-input')
Y = tf.placeholder(tf.float32, name='Y-input')

W1 = tf.Variable(tf.random_uniform([8, 8], -1.0, 1.0), name='Weight1')
# 9 hidden layers
W2 = tf.Variable(tf.random_uniform([8, 8], -1.0, 1.0), name='Weight2')
W3 = tf.Variable(tf.random_uniform([8, 8], -1.0, 1.0), name='Weight3')
W4 = tf.Variable(tf.random_uniform([8, 8], -1.0, 1.0), name='Weight4')
W5 = tf.Variable(tf.random_uniform([8, 8], -1.0, 1.0), name='Weight5')
W6 = tf.Variable(tf.random_uniform([8, 8], -1.0, 1.0), name='Weight6')
W7 = tf.Variable(tf.random_uniform([8, 8], -1.0, 1.0), name='Weight7')
W8 = tf.Variable(tf.random_uniform([8, 8], -1.0, 1.0), name='Weight8')
W9 = tf.Variable(tf.random_uniform([8, 8], -1.0, 1.0), name='Weight9')
W10 = tf.Variable(tf.random_uniform([8, 8], -1.0, 1.0), name='Weight10')

W11 = tf.Variable(tf.random_uniform([8, 1], -1.0, 1.0), name='Weight11')

b1 = tf.Variable(tf.zeros([8]), name="Bias1")
b2 = tf.Variable(tf.zeros([8]), name="Bias2")
b3 = tf.Variable(tf.zeros([8]), name="Bias3")
b4 = tf.Variable(tf.zeros([8]), name="Bias4")
b5 = tf.Variable(tf.zeros([8]), name="Bias5")
b6 = tf.Variable(tf.zeros([8]), name="Bias6")
b7 = tf.Variable(tf.zeros([8]), name="Bias7")
b8 = tf.Variable(tf.zeros([8]), name="Bias8")
b9 = tf.Variable(tf.zeros([8]), name="Bias9")
b10 = tf.Variable(tf.zeros([8]), name="Bias10")

b11 = tf.Variable(tf.zeros([1]), name="Bias11")


# Our hypothesis
with tf.name_scope("layer1") as scope:
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
with tf.name_scope("layer2") as scope:
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
with tf.name_scope("layer3") as scope:
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
with tf.name_scope("layer4") as scope:
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
with tf.name_scope("layer5") as scope:
    L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
with tf.name_scope("layer6") as scope:
    L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
with tf.name_scope("layer7") as scope:
    L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)
with tf.name_scope("layer8") as scope:
    L8 = tf.nn.relu(tf.matmul(L7, W8) + b8)
with tf.name_scope("layer9") as scope:
    L9 = tf.nn.relu(tf.matmul(L8, W9) + b9)
with tf.name_scope("layer10") as scope:
    L10 = tf.nn.relu(tf.matmul(L9, W10) + b10)
with tf.name_scope("last") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L10, W11) + b11)


# Cost function
with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
    cost_summ = tf.summary.scalar("cost", cost)

# Minimize
with tf.name_scope("train") as scope:
    a = tf.Variable(0.001) # Learning rate, alpha
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)



# Before starting, initialize the variables.
# We will `run` this first.
init = tf.global_variables_initializer()

#%%
# Launch the graph,
with tf.Session() as sess:
    # tensorboard --logdir=./logs/xor_logs
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs\\xor_logs", sess.graph)

    
    sess.run(init)
    # Fit the line.
    for step in range(50000):
        sess.run(train, feed_dict={X:x_data_training, Y:y_data_training})
        if step % 2000 == 0:
            summary = sess.run(merged, feed_dict={X:x_data_training, Y:y_data_training})
            writer.add_summary(summary, step)
            #print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2)
            print (step, sess.run(cost, feed_dict={X:x_data_training, Y:y_data_training}))

    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    y_data_pred = sess.run(tf.floor(hypothesis + 0.5),
                           feed_dict={X: x_data_testing, Y: y_data_testing})

    print (sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data_testing, Y:y_data_testing}))
    print ("Accuracy:", accuracy.eval({X:x_data_testing, Y:y_data_testing}))

#   print confusion_matrix(y_data_testing[:,0], y_data_pred[:,0], labels=[0, 1])
    pd_y_true = pd.Series(y_data_testing[:, 0])
    pd_x_pred = pd.Series(y_data_pred[:, 0])
    print (pd.crosstab(pd_y_true, pd_x_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    target_names = ['false', 'true']
    print (classification_report(y_data_testing[:, 0], y_data_pred[:, 0], target_names=target_names))
    print ('Precision', precision_score(y_data_testing[:, 0], y_data_pred[:, 0], average='binary',pos_label=1))


