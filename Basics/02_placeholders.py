"""
This file follows tutorial at : http://learningtensorflow.com/lesson4/
should visit : http://danijar.com/
cs224d
"""
# Placeholder : These are more basic structures in Just in time programming paradigm
#         -> A "placeholder" is simply a variable that will be assigned at a later stage.
#         -> It allows the creation of operation and build our computation graph without needing data
#         -> At the run time we then feed the data to these placeholders


import tensorflow as tf
# import numpy as np
x = tf.placeholder(tf.float32, shape=[None, 3], name='x')

y = x * 2

with tf.Session() as session:
    result = session.run(y, feed_dict={x : [[3, 4, 5],[1, 2, 3]]})
    print(result)
