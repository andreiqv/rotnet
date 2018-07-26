#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
in 
- conv 5х5 - pool 3х3 
- conv 4х4 - pool 3х3  
- conv 3х3 - pool 2х2  
- reshape - 1024 - dense -dense - mse

Validation:
train: 296.32 - 194.00
train: 326.57 - 356.00
train: 378.93 - 156.00
valid: 75.48 - 76.00
valid: 239.89 - 51.00
iteration 420: train_acc=0.2516, valid_acc=0.2653

v2.0 -  usage https://d4nst.github.io/2017/01/12/image-orientation/

"""

# export CUDA_VISIBLE_DEVICES=1

from __future__ import absolute_import,  division, print_function
import tensorflow as tf
from keras import backend as K


import sys
import math
import numpy as np
np.set_printoptions(precision=4, suppress=True)

#import load_data
import _pickle as pickle
import gzip

from rotate_images import *

BATCH_SIZE = 5
NUM_ITERS = 5000

data_file = "dump.gz"
f = gzip.open(data_file, 'rb')
data = pickle.load(f)
#data_1 = load_data(in_dir, img_size=(540,540))
#data = split_data(data1, ratio=(6,1,3))

train = data['train']
valid = data['valid']
test  = data['test']
print('train size:', train['size'])
print('valid size:', valid['size'])
print('test size:', test['size'])
im0 = train['images'][0]
print('Data was loaded.')
print(im0.shape)
#sys.exit()

#train['images'] = [np.transpose(t) for t in train['images']]
#valid['images'] = [np.transpose(t) for t in valid['images']]
#test['images'] = [np.transpose(t) for t in test['images']]
num_train_batches = train['size'] // BATCH_SIZE
num_valid_batches = valid['size'] // BATCH_SIZE
num_test_batches = test['size'] // BATCH_SIZE
print('num_train_batches:', num_train_batches)
print('num_valid_batches:', num_valid_batches)
print('num_test_batches:', num_test_batches)

SAMPLE_SIZE = train['size']
min_valid_accuracy = 1000


# some functions

def weight_variable(shape, name=None):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W, name=None):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)

def max_pool_2x2(x, name=None):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME',  name=name) 

def max_pool_3x3(x, name=None):
	return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME',  name=name)


def convPoolLayer(p_in, kernel, pool_size, num_in, num_out, func=None, name=''):
	W = weight_variable([kernel[0], kernel[1], num_in, num_out], name='W'+name)  # 32 features, 5x5
	b = bias_variable([num_out], name='b'+name)
	
	if func:
		h = func(conv2d(p_in, W, name='conv'+name) + b, name='relu'+name)
	else:
		h = conv2d(p_in, W, name='conv'+name) + b

	if pool_size == 2:
		p_out = max_pool_2x2(h, name='pool'+name)
	elif pool_size == 3:
		p_out = max_pool_3x3(h, name='pool'+name)
	else:
		raise("bad pool size")
	print('p{0} = {1}'.format(name, p_out))
	return p_out

def fullyConnectedLayer(p_in, input_size, num_neurons, func=None, name=''):
	num_neurons_6 = 128
	W = weight_variable([input_size, num_neurons], name='W'+name)
	b = bias_variable([num_neurons], name='b'+name)
	if func:
		h = func(tf.matmul(p_in, W) + b, name='relu'+name)
	else:
		h = tf.matmul(p_in, W) + b
	print('h{0} = {1}'.format(name, h))
	return h


# Create a new graph
graph = tf.Graph() # no necessiry

with graph.as_default():

	#K.set_learning_phase(1) #set learning phase
	K._LEARNING_PHASE = tf.constant(0)

	input_shape = (224, 224, 3)
	height, width, color = input_shape

	# 1. Construct a graph representing the model.
	x = tf.placeholder(tf.float32, [None, height, width, color]) # Placeholder for input.
	y = tf.placeholder(tf.float32, [None])   # Placeholder for labels.
	
	x_image = tf.reshape(x, [-1, height, width, color])
	#resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])

	resnet50 = tf.keras.applications.resnet50.ResNet50(
		weights='imagenet', include_top=False,
		input_shape=input_shape)

	output_resnet = resnet50(x_image)
	print('output_resnet =', output_resnet)

	flat = tf.reshape(output_resnet, [-1, 2048])
	f1 = fullyConnectedLayer(flat, input_size=2048, num_neurons=256, 
		func=tf.nn.relu, name='F1')
	f2 = fullyConnectedLayer(f1, input_size=256, num_neurons=1, 
		func=None, name='F2')	
	output = f2
	print('output =', output)

	# 2. Add nodes that represent the optimization algorithm.

	loss = tf.reduce_mean(tf.square(output - y))
	#loss = tf.reduce_mean(tf.squared_difference(y, output))
	#loss = tf.nn.l2_loss(output - y)
	#loss = tf.losses.mean_squared_error(labels=y, predictions=output)
	
	train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
	#train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
		
	#loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
	#train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
	#correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# 3. Execute the graph on batches of input data.
	with tf.Session() as sess:  # Connect to the TF runtime.
		init = tf.global_variables_initializer()
		sess.run(init)	# Randomly initialize weights.
		for iteration in range(NUM_ITERS):			  # Train iteratively for NUM_iterationS.		 

			if iteration % 200 == 0:

				#output_values = output.eval(feed_dict = {x:train['images'][:3]})
				#print('train: {0:.2f} - {1:.2f}'.format(output_values[0][0]*360, train['labels'][0]*360))
				#print('train: {0:.2f} - {1:.2f}'.format(output_values[1][0]*360, train['labels'][1]*360))
				output_values = output.eval(feed_dict = {x:valid['images'][:3]})
				print('valid: {0:.2f} - {1:.2f}'.format(output_values[0][0]*360, valid['labels'][0]*360))
				print('valid: {0:.2f} - {1:.2f}'.format(output_values[1][0]*360, valid['labels'][1]*360))
				#print('valid: {0:.2f} - {1:.2f}'.format(output_values[2][0]*360, valid['labels'][2]*360))

				output_angles_valid = []
				for i in range(num_valid_batches):
					feed_dict = {x:valid['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}
					print(feed_dict)
					output_values = output.eval(feed_dict=feed_dict)
					#print(i, output_values)
					#print(output_values.shape)
					t = [output_values[i][0]*360.0 for i in range(output_values.shape[0])]
					#print(t)
					output_angles_valid += t
				print(output_angles_valid)


			if iteration % 50 == 0:

				train_accuracy = np.mean( [loss.eval( \
					feed_dict={x:train['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
					y:train['labels'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
					for i in range(0,num_train_batches)])
				valid_accuracy = np.mean([ loss.eval( \
					feed_dict={x:valid['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
					y:valid['labels'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
					for i in range(0,num_valid_batches)])

				if valid_accuracy < min_valid_accuracy:
					min_valid_accuracy = valid_accuracy

				min_in_grad = math.sqrt(min_valid_accuracy) * 360.0
				print('iter {0:3}: train_loss={1:0.4f}, valid_loss={2:0.4f} (min={3:0.4f} ({4:0.2f} gr.))'.\
					format(iteration, train_accuracy, valid_accuracy, min_valid_accuracy, min_in_grad))

				"""
				#train_accuracy = loss.eval(feed_dict = {x:train['images'][0:BATCH_SIZE], y:train['labels'][0:BATCH_SIZE]})
				#valid_accuracy = loss.eval(feed_dict = {x:valid['images'][0:BATCH_SIZE], y:valid['labels'][0:BATCH_SIZE]})
				"""
			
			a1 = iteration*BATCH_SIZE % train['size']
			a2 = (iteration + 1)*BATCH_SIZE % train['size']
			x_data = train['images'][a1:a2]
			y_data = train['labels'][a1:a2]
			if len(x_data) <= 0: continue
			sess.run(train_op, {x: x_data, y: y_data})  # Perform one training iteration.		
			#print(a1, a2, y_data)			

		# Save the comp. graph

		x_data, y_data =  valid['images'], valid['labels'] #mnist.train.next_batch(BATCH_SIZE)		
		writer = tf.summary.FileWriter("output", sess.graph)
		print(sess.run(train_op, {x: x_data, y: y_data}))
		writer.close()  

		# Test of model
		"""
		HERE SOME ERROR ON GPU OCCURS
		num_test_batches = test['size'] // BATCH_SIZE
		test_accuracy = np.mean([ loss.eval( \
			feed_dict={x:test['images'][i*BATCH_SIZE : (i+1)*BATCH_SIZE]}) \
			for i in range(num_test_batches) ])
		print('Test of model')
		print('Test_accuracy={0:0.4f}'.format(test_accuracy))
		"""
		test_accuracy = loss.eval(feed_dict={x:test['images'][0:BATCH_SIZE]})
		print('Test_accuracy={0:0.4f}'.format(test_accuracy))				

		# Rotate images:
		in_dir = 'data'
		out_dir = 'valid'
		file_names = valid['filenames']
		angles = output_angles_valid
		rotate_images_with_angles(in_dir, out_dir, file_names, angles)
		
		"""
		# Saver
		saver = tf.train.Saver()		
		saver.save(sess, './save_model/my_test_model')  
		"""

