"""
This file follows the tutorial at : http://learningtensorflow.com/lesson2/
"""
# import tensorflow as tf
#
# x = tf.constant(35, name='x')
# y = tf.Variable(x+5, name='y')
#
# model = tf.global_variables_initializer()
# with tf.Session() as session:
#     session.run(model)
#     print(session.run(y))
#

#   Exercise : 1
"""
Simple working of the variable
"""
# import tensorflow as tf
#
# x = tf.constant([30, 47, 44], name='x')
# y = tf.Variable(x + 5, name='y')
#
# model = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(model)
#     print(sess.run(y))
#

#   Exercise : 2
"""
generating a variable with initial value with equation : ((5*(x**2)) - (3 * x) + 15)
"""
# import numpy as np
# import tensorflow as tf
#
# # generate random integers
# np.random.seed(1)
# arr = np.random.randint(1000, size=10)
#
# x = tf.constant(arr, name='x')
# y = tf.Variable(((5*(x**2)) - (3 * x) + 15), name='y')
#
# model = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(model)
#     print(sess.run(y))
#

#   Exercise : 3
"""
Manipulating the declared variable
"""
# import tensorflow as tf
#
# x = tf.Variable(0, name='x')
#
# model = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(model)
#     print("running in the loop")
#     for i in range(5):
#         x += 1
#         print(sess.run(x))

#   Exercise : 4
"""
This exercise is combination of exe 2 and 3
"""
# import tensorflow as tf
# import numpy as np
#
# avg = tf.Variable(0, name='avg')
#
# model = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(model)
#
#     for i in range(1000):
#         avg = ((avg * i) + np.random.randint(10000))/(i+1)
#
#     print(sess.run(avg))

# Exercise : 5
"""
This exercise is to get the tensorboard up and running
"""

import tensorflow as tf

x = tf.constant(35, name='x')
print(x)

y = tf.Variable(x + 5, name='y')

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./variables", sess.graph)
    model = tf.global_variables_initializer()
    sess.run(model)
    print(sess.run(y))

# Command to start tensorboard :
# tensorboard --logdir=/path/specified/in?filewriter