import sys
import binascii

import numpy as np
import json
from random import randint

from .CSVReader import CSVReader
from .StringManipulator import StringManipulator
from .CCNNConfig import CCNNConfig

class CCNNPreprocessor:
	config = CCNNConfig()
	max_character_in_article = config.model.max_letter_count
	
	# alphabets = config.model.alphabets
	# alphabets_length = len(alphabets)

	labels = config.model.labels
	labels_length = len(labels)

	@classmethod
	def normalize_content_data(cls, np_data, stemming=False, remove_stopword=False):
		if stemming:
			for i, dum in enumerate(np_data):
				print("Normalizing #{} data".format(i))
				for j, dum in enumerate(dum):
					np_data[i,j] = str(np_data[i,j])
					np_data[i,j] = StringManipulator.normalize_text(np_data[i,j], remove_stopword=remove_stopword)
		else:
			for i, dum in enumerate(np_data):
				print("Normalizing #{} data".format(i))
				for j, dum in enumerate(dum):
					np_data[i,j] = str(np_data[i,j])
					np_data[i,j] = np_data[i,j].lower()
					if remove_stopword:
						np_data[i,j] = StringManipulator.remove_stopwords(np_data[i,j])
	
		return np_data

	@classmethod
	def convert_dataset(cls, np_data, content_rows=[0, 1, 2], label_row=3, high_bit_dimension=False, convert_to_vector=True):
		# Merge content_rows and remove all non-content and non-label rows
		x_inputs = None
		y_labels = None

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
			if len(content) > cls.max_character_in_article:
				content = content[:cls.max_character_in_article] 
			else:
				content = content.ljust(cls.max_character_in_article) # Pad string to max_char_in_article

			# Convert content & label to vector
			content_vector = None
			label_vector = None

			label_eye = np.eye(cls.labels_length, dtype=int)

			# Create content vector representation
			if high_bit_dimension:
				# 69-bit
				for char in content:
					char_vector = np.zeros(len(cls.config.model.alphabets), dtype=np.int32)

					char_index = cls.alphabets.find(char)
					if char_index != -1: # If char is in the alphabet
						char_vector[char_index] = 1 # Set one-hot-vector

					if reverse: # Based on the paper, the content vector should be reversed
						char_vector = np.flip(char_vector, 0)

					char_vector = np.array([char_vector])

					if content_vector is not None:
						content_vector = np.append(content_vector, char_vector, axis=0)						
					else:
						content_vector = char_vector

			else:
				# 8-bit
				for char_v in cls.iter_bin(content):
					char_vector = np.array(list(map(int, char_v)))				
					if content_vector is not None:
						content_vector = np.append(content_vector, [char_vector], axis=0)						
					else:
						content_vector = np.array([char_vector])

			content_vector = content_vector.T 
			label_vector = label_eye[int(label)]

			if x_inputs is not None:
				x_inputs = np.append(x_inputs, [content_vector], axis=0)
				y_labels = np.append(y_labels, [label_vector], axis=0)
			else:
				x_inputs = np.array([content_vector])
				y_labels = np.array([label_vector])

		return x_inputs, y_labels

	@classmethod
	def iter_bin(cls, string):
		sb = string.encode('ascii')
		return (format(b, '08b') for b in sb)

	@classmethod
	def convert_content_data_to_vector(cls, np_data, alphabet, reverse=True):
		np_product = None
		for i, txt in enumerate(np_data):
			temp_np = None
			
			vec = np.array(cls.convert_str_to_vector(txt, alphabet))
			temp_np = np.array([vec])

			if np_product is not None:
				np_product = np.append(np_product, temp_np, axis=0)
			else:
				np_product = np.array(temp_np, dtype=object)

		print(np_product.shape)
		return np_product

	@classmethod
	def generate_vector_dictionary_from_string(cls, alphabet):
		length = len(alphabet)
		vector = None
		for index, char in enumerate(alphabet):

			v = np.zeros(length, dtype=np.int32)
			v[index] = 1
			v = np.array([v])
			
			if vector is not None:
				vector = np.append( (vector, v), axis=0)
			else:
				vector = np.array(v)

		return vector

	@classmethod
	def convert_str_to_vector(cls, char_seq, alphabet):
		int_seq = np.array([alphabet.find(char) for char in char_seq], dtype=np.int32)

		vector = np.zeros((cls.max_character_in_article, len(alphabet)), dtype=np.int32) 

		for index, i in enumerate(int_seq):
			if (i != -1):
				vector[index, i] = 1

		return vector

	@classmethod
	def shuffleData(cls, data_size):		
		np.random.seed(randint(0,300))

		shuffle_indices = np.random.permutation(np.arange(data_size))
		return shuffle_indices

	@classmethod
	def get_letter_count_information_from_article_list(cls, training_dir, content_rows=[1,2]):
		data = CSVReader.csv_to_numpy_list(training_dir)
		np_data = data[:, content_rows]
		np_data = cls.normalize_content_data(np_data)
		np_data = cls.merge_content_data(np_data)

		mx = -1
		mxi = -1

		mn = -1
		mni = -1

		mean = 0
		total = 0
		rows = 0

		for index, dum in enumerate(np_data):
			rows += 1
			length = len(dum)
			total += length

			if (mx == -1) or (mx < length):
				mx = length
				mxi = index
			if (mn == -1) or (mn > length):
				mn = length
				mni = index

		mean = total / rows

		print("Maximum character count: "+str(mx))
		print("Minimum character count: "+str(mn))
		print("Average character count: "+str(mean))

