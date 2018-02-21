# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:55:54 2018

@author: gihun
"""
import numpy as np
import tensorflow as tf
np.set_printoptions(linewidth=100,threshold=2000)

'''
basic concept and principle
@ graph and session
graph: define skeleton "cast" (only sturcture declaration) for "metal" (real memory in GPU)
        building graph is "complile process of the structure"
session: object managing "metal" (real memory in GPU)
        calling session.run is "runtime stage of the structure

@ there is 3 type of tensor object
1. placeholder : interface for CPU --> GPU memory copy. it serves as gateway from numpy value to tensor value
        placeholder is used for input. you must "feed" the input value (numpy value) to placebolder 
        when you evaluate tensor that depnds on the placeholder.
2. Variable : it holds its own GPU memory. it is not temporary.
        Variable is used as weights in network. It is the only we can save after training.
3. Tensor : it holds temporary GPU memory. tensor exists in order to represent intermediate result
        The type of all intermediate result is tensor.
ex)
    a : placeholder
    b : variable
    c = a+b  --> c is tensor     
'''
#%% must-know functions
#
tf.shape # tensor, variable's runtime shape
y.get_shape() # tensor, variable's compile time shape
y.get_shape().as_list()
#
tf.reduce_sum
tf.reduce_mean
tf.reduce_prod
#
tf.reshape
tf.tile
tf.concat
tf.stack
tf.unstack
#
tf.nn.relu
tf.nn.conv2d
tf.nn.conv2d_transpose
tf.nn.conv3d  
tf.nn.conv3d_transpose
tf.nn.max_pool
tf.nn.moments

#%% basic of tensorflow
# all the placeholder, variable, tensor will be contained by default_graph of tensorflow
tf.reset_default_graph() # reset all components in the graph like tensor, variable, placeholders.


tf.ones([3,3])
tf.zeros([3,3])
x = tf.placeholder(name="x",dtype=np.float32,shape=[3,3]) # placeholder
w = tf.get_variable(name="w",initializer=np.ones([3,3],dtype=np.float32)) #variable
y = x+w # tensor
y
# sess은 graph 선언을 끝낸다음에 인스턴스를 만드는것을 추천함.
sess = tf.Session() # session makes real GPU memory to implement graph
sess.run(tf.global_variables_initializer())
val = sess.run(y, feed_dict={x:np.ones([4,3],dtype=np.float32)})
print(val)
#%%
class Eval:
    def __init__(self):
        self.sess = tf.Session()
    def global_var_init(self):
        self.sess.run(tf.global_variables_initializer())
    def __call__(self,x,feed_dict=None):
        return self.sess.run(x,feed_dict)

#%% static shape vs dynamic shape (1)
# tf.shape(tensor): return tensor, variable's runtime shape. sess.run을 해야만 evaluation되는 shape
# tensor.get_shape(): return tensor, variable's compile time shape. sess.run을 하지 않고도 알수 있는 shape
tf.reset_default_graph()
x = tf.placeholder(name="x", dtype=tf.float32, shape=[None,3,3,1])
feed_dict = {
	x: np.ones([10,3,3,1], np.float32) # must np.float32 !
}
e = Eval()
e.global_var_init()
e(x,feed_dict)
print(x.get_shape().as_list()) # static shape: (1,3,3,1)
print(tf.shape(x)) # dynamic shape: (?,?,?,?)
print(e(tf.shape(x), feed_dict)) # dynamic shape (evaluated): (1,3,3,1)

#%% static shape vs dynamic shape (2)
tf.reset_default_graph()
x2 = tf.placeholder(name="x2", dtype=tf.float32, shape=[None,3,3,1])
feed_dict = {
	x2: np.ones([2,3,3,1], np.float32)
}
e = Eval()
print(x2.get_shape().as_list()) # static shape: (?,3,3,1)
print(tf.shape(x2)) # dynamic shape: (?,?,?,?)
print(e(tf.shape(x2), feed_dict)) # dynamic shape (evaluated): (2,3,3,1)

#%% static shape vs dynamic shape (3)
tf.reset_default_graph()
x3 = tf.placeholder(name="x3", dtype=tf.float32, shape=[None,None,None,None])
feed_dict = {
	x3: np.ones([4,10,10,3], np.float32)
}
e = Eval()
print(x3.get_shape().as_list()) # static shape: (?,?,?,?)
print(tf.shape(x3)) # dynamic shape: (?,?,?,?)
print(e(tf.shape(x3), feed_dict)) # dynamic shape (evaluated): (4,10,10,3)

#%% reduce functions
# tf.reduce_* 형태를 띄는 함수들
tf.reset_default_graph()
x = tf.placeholder(name="x",dtype=tf.float32,shape=[3,3,3,1])
x_ = np.reshape(np.arange(0,27,1,np.float32),[3,3,3,1])
feed_dict = {x:x_}
e = Eval()
print(e(tf.reduce_sum(x,reduction_indices=[1,2]),feed_dict))
print(e(tf.reduce_sum(x),feed_dict))
print(np.sum(x_))
print(e(tf.reduce_sum(x,axis=0),feed_dict))
print(np.sum(x_,axis=0))
print(e(tf.reduce_mean(x),feed_dict))
print(np.mean(x_))
print(e(tf.reduce_mean(x,axis=0,keep_dims=True),feed_dict))
print(np.mean(x_,axis=0,keepdims=True))
print(e(tf.reduce_prod(x,axis=0),feed_dict))
print(np.prod(x_,axis=0))

#%% dimension related functions
tf.reset_default_graph()
x = tf.placeholder(name="x", dtype=tf.float32, shape=[3,3])
x_ = np.reshape(np.arange(0,9,1,np.float32), [3,3])
feed_dict = {	x: x_}
e = Eval()
print(x_)
print(e(x,feed_dict))
print(e(tf.reshape(x, [9]), feed_dict))
print(np.reshape(x_, [9]))
print(e(tf.tile(x, [2,2]), feed_dict))
print(tf.tile(x,[2,2]).get_shape().as_list())
print(np.tile(x_, [2,2]))

#%% stack, concat, unstack, split,  functions
tf.reset_default_graph()
x = tf.placeholder(name="x", dtype=tf.float32, shape=[3,3])
y = tf.placeholder(name="x", dtype=tf.float32, shape=[3,3])
x_ = np.reshape(np.arange(0,9,1,np.float32), [3,3])
y_ = np.reshape(np.arange(0,9,1,np.float32), [3,3])
feed_dict = {
	x: x_,
	y: y_
}
e = Eval()
print(x_)
print(y_)
print(e(tf.concat([x,y], axis=1), feed_dict))
print(np.concatenate([x_,y_], axis=1))
print(e(tf.stack([x,y], axis=0), feed_dict))
print(np.stack([x_,y_], axis=0))
print(e(tf.unstack(x, axis=0), feed_dict))
for a in e(tf.unstack(x, axis=0),feed_dict):
    print(a)
print(e(tf.split(x, 3, axis=0), feed_dict))
for a in e(tf.split(x, 3, axis=0),feed_dict):
    print(a)
print(np.split(x_, [1,2], axis=0))









#%% deep learning layers
# here we see: relu, conv2d usage.
# you must also be accustomed to how to use conv2d_transpose, conv3d, conv3d_transpose
tf.reset_default_graph()
x = tf.placeholder(name="x", dtype=tf.float32, shape=[None,None,None,1]) # (b,y,x,c)
w = tf.placeholder(name="w", dtype=tf.float32, shape=[3,3,1,3]) # (y,x,cin,cout)
b = tf.placeholder(name="b", dtype=tf.float32, shape=[3]) # (cout)
conv = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding="SAME", name="conv") # (b,y,x,cout)
# x: (b,y,x,c)
# w: (y,x,cin,cout)
# strides = [1, y, x, 1]
# padding = "SAME"(zero padding), "VALID"(edge pixels will be removed)
add = conv + b
relu = tf.nn.relu(add, name="relu")
pool = tf.nn.max_pool(relu,[1,2,2,1],strides=[1,2,2,1],padding='SAME', name="pool")
e = Eval()
x_ = np.tile(-np.identity(3, np.float32), [3,3])[None,:,:,None]
x_.shape
w_ = np.array([[1,1,1], [0,1,0], [0,0,0]], np.float32)[:,:,None,None]
w_ = np.concatenate([w_]*3, axis=3)
w_.shape
feed_dict = {
	x: x_,
	w: w_,
	b: np.array([1.5]*3, np.float32)
}
print(x_[0,:,:,0])
print(w_[:,:,0,0])
print(e(tf.shape(x),feed_dict))
print(e(conv, feed_dict)[0,:,:,0])
print(e(add, feed_dict)[0,:,:,0])
print(e(relu, feed_dict)[0,:,:,0])
print(e(pool, feed_dict)[0,:,:,0])

print(relu.get_shape())
print(e(tf.shape(relu),feed_dict))
print(e(tf.shape(pool),feed_dict))

print(pool.get_shape())



#%% deep learning layers full layers
tf.reset_default_graph()
x = tf.placeholder(name="x", dtype=tf.float32, shape=[None,10])
w = tf.placeholder(name="w", dtype=tf.float32, shape=[10,10])
b = tf.placeholder(name="b", dtype=tf.float32, shape=[10])
full = tf.add(tf.matmul(x, w),b, name="full")
x_ = np.ones([3,10], np.float32)
w_ = np.ones([10,10], np.float32)
feed_dict = {
	x: x_,
	w: w_,
     b: [0.5]*10
}
e = Eval()
print(x_)
print(w_)
print(e(full, feed_dict))







#%% conv  series
tf.reset_default_graph()
x = tf.placeholder(name="x",dtype=tf.float32, shape=[None,None,None,1])
w = tf.placeholder(name="w",dtype=tf.float32, shape=[3,3,3,1])
b = tf.placeholder(name="b",dtype=tf.float32, shape=[3])
conv = tf.nn.conv2d_transpose(x,w, strides=[1,1,1,1],output_shape=[1,5,5,3],padding="SAME", name='conv')
add = conv+b
relu = tf.nn.relu(add,name="relu")
e = Eval()
x_ = -np.identity(5, np.float32)[None,:,:,None]
w_ = np.array([[0,0,0],[0,0,1],[0,0,0]],np.float32)[:,:,None,None]
w_ = np.concatenate([w_]*3, axis=2)
feed_dict = {x:x_,w:w_,b:np.array([0.5]*3,np.float32)}
print(x_[0,:,:,0])
print(w_[:,:,0,0])
print(e(conv, feed_dict)[0,:,:,0])
print(e(add, feed_dict)[0,:,:,0])
print(e(relu, feed_dict)[0,:,:,0])
'''
conv2d_transpose needs output_shape and it should be static shape.
so I recomend you not to use conv2d_transpose, but conv2d.
'''
tf.reset_default_graph()
x3 = tf.placeholder(name="x3", dtype=tf.float32, shape=[None,None,None,None,1])
w3 = tf.placeholder(name="w3", dtype=tf.float32, shape=[3,3,3,1,3])
b3 = tf.placeholder(name="b3", dtype=tf.float32, shape=[3])
conv3d = tf.nn.conv3d(x3,w3,strides=[1,1,1,1,1],padding="SAME",name="conv3d")
add = conv3d+b3
relu = tf.nn.relu(add,name="relu")

e=Eval()

x_ = -np.zeros([5,5,5])[None,:,:,:,None]
for i in range(5):
    x_[0,i,i,i,0]=-1


w_ = np.zeros([3,3,3])[:,:,:,None,None]
w_[1,1,1,0,0]=1
w_[0,1,1,0,0]=1
w_ = np.concatenate([w_]*3, axis=4)

feed_dict= {x3:x_, w3:w_,b3:np.array([0.5]*3,np.float32)}
print(x_[0,:,:,:,0])
print(w_[:,:,:,0,0])
print(e(conv3d, feed_dict)[0,:,:,:,0])
print(e(add,feed_dict)[0,:,:,:,0])
print(e(relu, feed_dict)[0,:,:,:,0])






#%% variable and placeholder
tf.reset_default_graph()
x = tf.Variable(np.ones([3,3],dtype=np.float32))

e=Eval()
e.global_var_init()
e(x)

'''
variable must initialize! if not, error : Attemptimg to use uninitialized value Variable
On the other hand, in case of placeholder, we must use it with feed_dict(like initializing) 
'''
####################
#%% must-know functions
#
tf.shape # tensor, variable's runtime shape
y.get_shape() # tensor, variable's compile time shape
y.get_shape().as_list()
#
tf.reduce_sum
tf.reduce_mean
tf.reduce_prod
#
tf.reshape
tf.tile
tf.concat
tf.stack
tf.unstack
#
tf.nn.relu
tf.nn.conv2d
tf.nn.conv2d_transpose
tf.nn.conv3d  
tf.nn.conv3d_transpose
tf.nn.max_pool
tf.nn.moments


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
