# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 13:23:07 2017
Chapter 5
basic of tensorflow
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
#%% basic of tensorflow

# all the placeholder, variable, tensor will be contained by default_graph of tensorflow
tf.reset_default_graph() # reset all components in the graph like tensor, variable, placeholders.
tf.ones([3,3])
tf.zeros([3,3])
x = tf.placeholder(name="x",dtype=np.float32,shape=[3,3]) # placeholder
w = tf.get_variable(name="w",initializer=np.ones([3,3],dtype=np.float32)) #variable
y = x+w # tensor


# sess은 graph 선언을 끝낸다음에 인스턴스를 만드는것을 추천함.
sess = tf.Session() # session makes real GPU memory to implement graph
sess.run(tf.global_variables_initializer())
val = sess.run(y, feed_dict={x:np.ones([3,3],dtype=np.float32)})
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
x = tf.placeholder(name="x", dtype=tf.float32, shape=[1,3,3,1])
feed_dict = {
	x: np.ones([1,3,3,1], np.float32) # must np.float32 !
}
e = Eval()
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
print(pool.get_shape())
print(e(tf.shape(pool),feed_dict))




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

