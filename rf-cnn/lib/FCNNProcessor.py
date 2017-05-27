import json
import os
import time
import datetime

import sklearn
import sklearn.metrics
import tensorflow as tf
import numpy as np

from sklearn.cross_validation import KFold

from .CSVReader import CSVReader
from .FCNNModel import FCNNModel
from .FCNNPreprocessor import FCNNPreprocessor
from .FCNNConfig import FCNNConfig
from .ListManipulator import ListManipulator
from .DurationRecorder import DurationRecorder

class FCNNProcessor:

	config = FCNNConfig()

	@classmethod
	def train_step(cls, x_batch, y_batch, cnn, session, train_op, global_step):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: cls.config.training.drop_out
            }

            _, step, loss, accuracy, predictions= session.run(
                [train_op, # Updates parameter of the network
                 global_step, 
                 cnn.loss, 
                 cnn.accuracy,
                 cnn.predictions],
                feed_dict)

            batch_labels = np.argmax(y_batch, 1)
            precision = sklearn.metrics.precision_score(batch_labels, predictions)
            recall = sklearn.metrics.recall_score(batch_labels, predictions)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {}, acc {}, precision {}, recall: {}".format(time_str, step, loss, 
            																	  accuracy, precision, recall))

            return time_str, step, loss, accuracy, precision, recall

	@classmethod
	def dev_step(cls, x_batch, y_batch, cnn, session, global_step):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0 # Disables learning
            }
            step, loss, accuracy, predictions = session.run(
                [global_step,
                 cnn.loss, 
                 cnn.accuracy,
                 cnn.predictions],
                feed_dict)

            batch_labels = np.argmax(y_batch, 1)
            precision = sklearn.metrics.precision_score(batch_labels, predictions)
            recall = sklearn.metrics.recall_score(batch_labels, predictions)     

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {}, acc {}, precision {}, recall: {}".format(time_str, step, loss, 
            																	  accuracy, precision, recall))

            return time_str, step, loss, accuracy, precision, recall		

	@classmethod
	def initial_train(cls, training_dir, content_rows=[1,2], label_row=3, ratio=0.8):
		with tf.Graph().as_default():
			session_conf = tf.ConfigProto(
				allow_soft_placement=True,
				log_device_placement=False) #Let's tweak this later

			session = tf.Session(config=session_conf)
			
			with session.as_default():
				cnn = FCNNModel()

				# Define Training Procedure
				global_step = tf.Variable(0, name="global_step", trainable="false")
				optimizer = tf.train.AdamOptimizer(cls.config.model.th)
				grads_and_vars = optimizer.compute_gradients(cnn.loss)
				train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			# No need to create summaries atm

			# Initialize All Variables
			curr_time = None

			print("Reading Data")
			data = CSVReader.csv_to_numpy_list(training_dir) # 2D-Array (Elements, Variables)
			print("Finished Reading")

			data = FCNNPreprocessor.normalize_content_data(data)
			input_data, label_data = FCNNPreprocessor.convert_dataset(data)

			kf = KFold(len(input_data), n_folds=10, shuffle=True)
			DurationRecorder.start_log()

			index = 0

			for train_indices, test_indices in kf:		
				index += 1
				session.run(tf.global_variables_initializer())

				for epoch in range(cls.config.training.num_of_epoches):

					################ Training ###############
					x_training_batch_data = np.array(input_data[train_indices])
					y_training_batch_data = np.array(label_data[train_indices])

					time_str, step, loss, accuracy, precision, recall = cls.train_step(x_training_batch_data, y_training_batch_data, 
									 								cnn, session, train_op, global_step)

					current_step = tf.train.global_step(session, global_step)

				################ Testing ################

				x_test_batch_data = np.array(input_data[test_indices])
				y_test_batch_data = np.array(label_data[test_indices])
				
				time_str, step, loss, accuracy, precision, recall = cls.dev_step(x_test_batch_data, y_test_batch_data, 
																		cnn, session, global_step)
				
				DurationRecorder.pr_epoch_plotter(index, precision, recall)
				DurationRecorder.pr_epoch_logger(index, precision, recall)	

	@classmethod
	def training(cls, training_dir="./small_data.csv", content_rows=[1,2], label_row=3):
		training_data = CSVReader.csv_to_numpy_list(training_dir)

		# Preprocess Test Data
		content_data = training_data[:, content_rows]
		label_data = training_data[:, label_row]
		content_data = FCNNPreprocessor.normalize_content_data(content_data)
		content_data = FCNNPreprocessor.merge_content_data(content_data)
		# content_vector = FCNNPreprocessor.convert_content_data_to_vector(content_data, cls.alphabet)

		print("Begin Training...")
		


