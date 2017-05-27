import sys
from datetime import datetime
import re

from random import randint
import numpy as np
import json
import fasttext

from .CSVReader import CSVReader
from .ListManipulator import ListManipulator
from .StringManipulator import StringManipulator
from .FCNNConfig import FCNNConfig

class FCNNPreprocessor:
	config = FCNNConfig()
	max_word_count = config.model.max_word_count

	@classmethod
	def get_content_data(cls, np_data, content_rows=[1,2]):
		data =  np_data[:, content_rows]
		# new_data = None

		# for dum in data:
		# 	row = ''
		# 	for dum in dum:
		# 		row += dum

		# 	if new_data is not None:
		# 		new_data = np.append(new_data, [row], axis=0)
		# 	else:
		# 		new_data = np.array([row])

		return data

	@classmethod
	def normalize_content_data(cls, np_data, stemming=False, remove_stopword=False):
		if stemming:
			for i, dum in enumerate(np_data):
				print("Normalizing #{}".format(i))
				for j, dum in enumerate(dum):
					np_data[i,j] = str(np_data[i,j])
					np_data[i,j] = StringManipulator.normalize_text(np_data[i,j], remove_stopword=remove_stopword)
		else:
			for i, dum in enumerate(np_data):
				print("Normalizing #{}".format(i))
				for j, dum in enumerate(dum):
					np_data[i,j] = str(np_data[i,j])
					np_data[i,j] = np_data[i,j].lower()
					if remove_stopword:
						np_data[i,j] = StringManipulator.remove_stopwords(np_data[i,j])
	
		return np_data

	@classmethod
	def convert_dataset(cls, np_data, content_rows=[1,2], label_row=3, reverse=True, convert_to_vector=True):
		# Merge content_rows and remove all non-content and non-label rows
		x_inputs = None
		y_labels = None
		print(cls.config.word_model.model_dir)
		model = fasttext.load_model(cls.config.word_model.model_dir)

		for i, rows in enumerate(np_data):
			print("Merging #{}".format(i))
			content = ''
			label = None

			# Combine content rows
			for j, column in enumerate(rows): 
				if(j in content_rows): 
					content += column # Merge Selected Contents
				elif(j == label_row):
					label = column # Retrieve Label

			# Limit Letter Count
			if len(content) > cls.max_word_count:
				content = content[:cls.max_word_count] 
			else:
				content = content.ljust(cls.max_word_count) # Pad string to max_char_in_article

			# Convert content & label to vector
			content_vector = None
			label_vector = None

			for word in content:
				temp_vec = np.array(model[word])				
				if content_vector is not None:
					content_vector = np.append(content_vector, [temp_vec], axis=0)
				else:
					content_vector = np.array([temp_vec])
			content_vector = content_vector.T

			label_eye = np.eye(cls.config.model.num_of_classes, dtype=int)
			label_vector = label_eye[int(label)]

			if x_inputs is not None:
				x_inputs = np.append(x_inputs, [content_vector], axis=0)
				y_labels = np.append(y_labels, [label_vector], axis=0)
			else:
				x_inputs = np.array([content_vector])
				y_labels = np.array([label_vector])

			# print("Input Data: {}".format(content_vector.shape))
			# print("Label Data: {}".format(label_vector.shape))

		return x_inputs, y_labels



	@classmethod
	def shuffleData(cls, data_size):
		np.random.seed(randint(0,300))

		shuffle_indices = np.random.permutation(np.arange(data_size))
		return shuffle_indices

	@classmethod
	def get_word_count_information_from_article_list(cls, training_dir, content_rows=[0, 1, 2]):
		np_data = CSVReader.csv_to_numpy_list(training_dir)		
		np_data = ListManipulator.merge_content_data(np_data)

		mx = -1
		mxi = -1

		mn = -1
		mni = -1

		mean = 0
		total = 0
		rows = 0

		for index, dum in enumerate(np_data):
			rows += 1
			length = len(re.compile("[\W]+").split(dum))
			total += length

			if (mx == -1) or (mx < length):
				mx = length
				mxi = index
			if (mn == -1) or (mn > length):
				mn = length
				mni = index

		mean = total / rows

		print("Maximum word count: "+str(mx))
		print("Minimum word count: "+str(mn))
		print("Average word count: "+str(mean))

