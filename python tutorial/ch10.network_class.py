# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:46:06 2017

@author: gihun
"""
import tensorflow as tf
import numpy as np
#%%

class struct:
	#http://stackoverflow.com/questions/1325673/how-to-add-property-to-a-class-dynamically @find=pass
	#https://www.tutorialspoint.com/python/python_pass_statement.htm
	#http://stackoverflow.com/questions/5969806/print-all-properties-of-a-python-class
	#http://stackoverflow.com/questions/1436703/difference-between-str-and-repr-in-python
	def __str__(self):
		return "struct{\n  " +  "\n  ".join(["{}: {}".format(k,v) for k,v in vars(self).items()]) + "\n}"
	def __repr__ (self):
		print("struct{\n  " +  "\n  ".join(["{}: {}".format(k,v) for k,v in vars(self).items()]) + "\n}")
	pass
#%%
class Layer:
    def weight_variable(shape, name):
        return tf.get_variable(name=name, dtype=tf.float32,shape=shape ,initializer=tf.contrib.layers.xavier_initializer_conv2d())
    def bias_variable(shape, name):
        return tf.get_variable(name=name, dtype=tf.float32, shape=shape, initializer=tf.constant_initializer(0))
        
    def full_layer(x, W_shape):
        W = Layer.weight_variable(W_shape, "W")
        b = Layer.bias_variable(W_shape[-1], "b")
        full = tf.add(tf.matmul(x,W),b,name="full")
        return full
        
    def conv_layer(x, param_bn, W_shape, stirdes=[1,1,1,1], padding="SAME"):
        W = Layer.weight_variable(W_shape, "W")
        b = Layer.bias_variable(W_shape[-1],"b")
        conv = tf.add(tf.nn.conv2d(x, W, strides=stirdes, padding=padding), b, name="conv")
        with tf.variable_scope("bn"):
            bn = Layer.batch_norm(conv, W_shape[-1], param_bn)      
        relu = tf.nn.relu(bn, name="relu")
        return relu
        
    def batch_norm(x, n_out, phase_train, name, decay=0.99):
        with tf.variable_scope(name):
            beta = tf.get_variable(initializer=tf.constant(0.0, shape=[n_out]), name="beta")
            gamma = tf.get_variable(initializer=tf.constant(1.0, shape=[n_out]), name="gamma")
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name="moments")
            mean_sh = tf.get_variable(initializer=tf.zeros([n_out]), name="mean_sh", trainable=False)
            var_sh = tf.get_variable(initializer=tf.ones([n_out]), name="var_sh", trainable=False)
            def mean_var_with_update():
                mean_assign_op = tf.assign(mean_sh, mean_sh*decay+ (1-decay)*batch_mean)
                var_assign_op = tf.assign(var_sh, var_sh*decay+(1-decay)*batch_var)
                with tf.control_dependencies([mean_assign_op, var_assign_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(tf.cast(phase_train, tf.bool), mean_var_with_update, lambda:(mean_sh, var_sh))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3, name="normed")
        return normed

class pracnet11:
    def __init__(self, name="pracnet11", channel_in=128):
        self.name = name
        self.channel_in = channel_in  
    def conv3d_layer(x, W_shape, strides=[1,1,1,1,1], padding="SAME"):
        w = Layer.weight_variable(W_shape, "w")
        b = Layer.bias_variable(W_shape[4], "b")
        conv = tf.add(tf.nn.conv3d(x, w, strides=strides, padding=padding), b, name="conv")
        return conv
    def block_layer(x, W_shape, param_bn=1, param_bn_decay=0.99, strides=[1,1,1,1,1], padding="SAME"):
        '''
        {bn, relu, conv} * 3
        '''
        out = x
        for i in range(3):
            out = tf.nn.relu(out, name="relu{}".format(i))
            with tf.variable_scope("conv{}".format(i)):
                out = pracnet11.conv3d_layer(x, W_shape, strides=strides, padding=padding)
        return out
    def __call__(self, x, param_bn, param_bn_decay=0.99, reuse=True):
        old_vars = tf.get_collection(key='variablse')
        with tf.variable_scope(self.name, reuse=reuse):
            with tf.variable_scope("conv_start"):
                out = pracnet11.conv3d_layer(x, [3,3,3,3,64])
            for i in range(3):
                with tf.variable_scope("block{}".format(i)):
                    out = pracnet11.block_layer(out,[3,3,3,64,64])
            with tf.variable_scope("conv_end"):
                out = pracnet11.conv3d_layer(out, [3,3,3,64,3])
        cur_vars = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES)
        try:                  
            for var in cur_vars:
                if var not in old_vars:
                    tf.add_to_collection(self.name+"/variables", var)
        except Exception as e:
            print(e)
        return out
    
    @property
    def variables(self):
        return tf.get_collection(self.name+"/variables")
        
#%%
tf.reset_default_graph()
x =tf.placeholder(dtype=tf.float32,shape=[None,32,32,32,3])
label = tf.placeholder(dtype=tf.float32, shape=[None,32,32,32])
param = struct()
param.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name="param.leraning_rate")
param.bn = tf.placeholder(dtype=tf.float32, shape=[], name="param.bn")
param.bn_decay =tf.placeholder(dtype=tf.float32, shape=[], name="param.bn_decay")

Prac11 = struct() 
Prac11.net = pracnet11(name="Prac11", channel_in=3)
Prac11.output = Prac11.net(x, param.bn, param.bn_decay, reuse=False)
Prac11.saver = tf.train.Saver(Prac11.net.variables, max_to_keep=500)

 
#%% batch_norm

def batch_norm(x, n_out, phase_train, name, decay=0.99):
    beta = tf.get_variable(initializer=tf.constant(0.0, shape=[n_out]), name="beta")
    gamma = tf.get_variable(initializer=tf.constant(1.0, shape=[n_out]), name="gamma")
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name="moments")

    mean_sh = tf.get_variable(initializer=tf.zeros([n_out]), name="mean_sh", trainable=False)
    var_sh = tf.get_variable(initializer=tf.ones([n_out]), name="var_sh", trainable=False)
    mean_sh = mean_sh*decay+(1-decay)*batch_mean
#%% assign examples!

tf.reset_default_graph()
x = tf.get_variable(initializer=tf.ones([3]), name="x", trainable=False)
y = tf.get_variable(initializer=tf.ones([3]), name="y")

def mv_update():
    x = x+y
    return x
mv_update()
def mv_update_assign():
    x_assign_op = tf.assign(x, x+y)
    with tf.control_dependencies([x_assign_op]):
        return tf.identity(x) #거지같은 문법
        
x2 = mv_update_assign()
    
sess = tf.Session()
sess.run(tf.global_variables_initializer())
x2 = mv_update_assign()
sess.run(x2)
#%%
tf.reset_default_graph()
x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x, 1)

with tf.control_dependencies([x_plus_1]):
    y = x
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        print(y.eval())
#%%     
tf.reset_default_graph()
x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x, 1)

with tf.control_dependencies([x_plus_1]):
    y = tf.identity(x)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        print(y.eval())

#%%
b = Layer.bias_variable([3],"b")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))

#%% structure



