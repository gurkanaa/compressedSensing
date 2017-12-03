##python tests for first hand
#read file
import tensorflow as tf
import numpy as np
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
sess1=tf.Session()
node3=tf.add(node1,node2)
W=tf.Variable([.3],dtype=tf.float32)
b=tf.Variable([-.3],dtype=tf.float32)
x=tf.placeholder(tf.float32)
linear_model=W*x+b
init=tf.global_variables_initializer()
sess1.run(init)
y=tf.placeholder(tf.float32)
squared_error=tf.square(linear_model-y)
loss=tf.reduce_sum(squared_error)
fixW=tf.assign(W,[-1.])
fixb=tf.assign(b,[1.])
sess1.run([fixW,fixb])
sess1.run(init)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for i in range(1000):
  sess1.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess1.run([W,b]))
