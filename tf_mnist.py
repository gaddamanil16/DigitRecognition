# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:04:53 2019

@author: gadda
"""

import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)

#%%
#learning params
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

#%%
#input graph
x = tf.placeholder("float", [None,784]) #image of dimension 28*28
y = tf.placeholder("float",[None,10]) #labels one hot encoded

#Model

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#Scope 1
with tf.name_scope("Wx_b") as scope:
    #linear model
    model = tf.nn.softmax(tf.matmul(x,W) + b)

#summary
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)

#Scope 2
with tf.name_scope("cost_function") as scope:
    #Minimize error using cross entropy
    #cross entropy
    cost_function = -tf.reduce_sum(y*tf.log(model))
    tf.summary.scalar("cost_function", cost_function)

#Scope3
with tf.name_scope("train") as scope:
    #gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)
init = tf.initializers.global_variables()

#merging summaries

merged_summary_op = tf.summary.merge([w_h, b_h])

#%%

with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter('data/logs', graph_def = sess.graph_def)

    #Training
    for iteration in range(training_iteration):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            sess.run(optimizer, feed_dict = {x: batch_xs, y:batch_ys})

            avg_cost += sess.run(cost_function, feed_dict = {x: batch_xs, y:batch_ys})/total_batch

            summary_str = sess.run(merged_summary_op, feed_dict = {x:batch_xs, y:batch_ys})

            summary_writer.add_summary(summary_str, total_batch + i)

        if iteration % display_step == 0:
            print("Iteration:", '%04d' %(iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Tuning completed")

    summary_writer.flush()
    summary_writer.close()
    predictions = tf.equal(tf.argmax(model,1), tf.argmax(y,1))

    accuracy = tf.reduce_mean(tf.cast(predictions,"float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))

#%%

