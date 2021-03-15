import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

def load(dir_name):
	"""
	Loads data from directory and makes tf.data.Dataset objects for train, test and validate
	:return train, test, validate: tf.data.Dataset
	"""
	DATASET_SIZE = 6615     # needed to split data
	# get the dataset into td.data.Dataset
	dataset = tf.keras.preprocessing.image_dataset_from_directory(dir_name, image_size=(640, 480))
	train_size = int(0.6 * DATASET_SIZE)
	val_size = int(0.2 * DATASET_SIZE)
	test_size = int(0.2 * DATASET_SIZE)

	test_dataset = dataset.take(test_size)
	train_dataset = dataset.skip(test_size)
	val_dataset = test_dataset.take(val_size)
	test_dataset = test_dataset.skip(val_size)




def split(data, train_size=0.6, test_size=0.2, validate_size=0.2):
	"""
	Splits data into 3 sets: train, test, and validate
	:param test_size: float [0,1], percentage of test set
	:param train_size: float [0,1], percentage of train set
	:param validate_size: float [0,1], percentage of validation set
	:param data: pandas.df, columns Label, Path
	:return X_train, y_train, X_test, y_test, X_validate, y_validate: np.arrays
	"""
	if train_size + test_size + validate_size != 1:
		raise ValueError("Sizes don't add up to 1 in data splitting")
	X_train, y_train, X_test, y_test = train_test_split(data, test_size=test_size)
	# calculate ratio of validation set and whats left after separating test set
	validate_ratio = validate_size / (train_size + validate_size)
	X_train, y_train, X_validate, y_validate = train_test_split(X_train, y_train, test_size=validate_ratio)
	return X_train, y_train, X_test, y_test, X_validate, y_validate


if __name__ == '__main__':
	load('Soda_Bottles/')