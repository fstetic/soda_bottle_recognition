import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load(dir_name):
	"""
	Loads data from directory and makes tf.data.Dataset objects for train, test and validate
	:param dir_name: string, name of the dataset directory
	:return train, test, validate: tf.data.Dataset
	"""
	# get csv
	df = pd.read_csv(dir_name + 'full.csv')
	# split df (split path-label)
	train, test, validate = split(df)
	# load each df
	# https://stackoverflow.com/questions/63761717/load-image-dataset
	# Data augmentation pipeline
	# using rotation for 90 degrees, shifting 20% on any side, changing brightness by 0.5 up or down, and zooming in or out 30%
	train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=90, width_shift_range=0.2,
								height_shift_range=0.2, brightness_range=(0.5, 1.5), zoom_range=0.3)
	test_validate_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

	# Reading files from path in df
	train_ds = train_datagen.flow_from_dataframe(train, directory=dir_name, x_col='Path', y_col='Label')
	test_ds = test_validate_datagen.flow_from_dataframe(test, directory=dir_name, x_col='Path', y_col='Label')
	validate_ds = test_validate_datagen.flow_from_dataframe(validate, directory=dir_name, x_col='Path', y_col='Label')
	return train_ds, test_ds, validate_ds


def split(data, train_size=0.6, test_size=0.2, validate_size=0.2):
	"""
	Splits data into 3 sets: train, test, and validate
	:param test_size: float [0,1], percentage of test set
	:param train_size: float [0,1], percentage of train set
	:param validate_size: float [0,1], percentage of validation set
	:param data: pandas.df, columns Label, Path
	:return train, test, validate: pandas.df
	"""
	if train_size + test_size + validate_size != 1:
		raise ValueError("Sizes don't add up to 1 in data splitting")
	train, test = train_test_split(data, test_size=test_size)
	# calculate ratio of validation set and whats left after separating test set
	validate_ratio = validate_size / (train_size + validate_size)
	train, validate = train_test_split(train, test_size=validate_ratio)
	return train, test, validate