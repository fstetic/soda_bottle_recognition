import tensorflow as tf


def custom_model_train(train_ds, test_ds, validate_ds):
	"""
	:param train_ds: tf.DataframeIterator
	:param test_ds: tf.DataframeIterator
	:param validate_ds: tf.DataframeIterator
	:return: tf.model
	"""
	# stack of layers
	model = tf.keras.models.Sequential()
	# 32 feature maps made with 5x5 kernel over input image
	model.add(tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(256, 256, 3)))
	# pooling to reduce dimensions in half, left with 126x126x32
	model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
	model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	# dimensionality reduction, getting 32 feature maps from those 64 from previous conv layer
	# combats overfitting and reduces number of parameters
	model.add(tf.keras.layers.Conv2D(32, (1, 1), activation='relu'))
	# pooling to reduce dimensions in half, left with 62x62x32
	model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
	model.add(tf.keras.layers.Flatten())
	# dropout with 0.3% chance to combat overfitting
	model.add(tf.keras.layers.Dropout(0.3))
	# fully connected layer that makes sense of the features extracted by previous layers
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	# softmax layer for classification
	model.add(tf.keras.layers.Dense(8, activation='softmax'))

	# stop when validation loss doesn't decrease more than 0.01 in 3 consecutive epochs
	callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.01)

	model.compile(optimizer='adam',
				loss='categorical_crossentropy',
				metrics=['accuracy'])

	model.fit(train_ds, epochs=40, validation_data=validate_ds, callbacks=[callback])

	print(model.evaluate(test_ds))

	return model


def tl_model_train(train_ds, test_ds, validate_ds):
	"""
	I'll be using InceptionV3 because it has good accuracies with acceptable number of parameters https://keras.io/api/applications/
	It will be used as a feature extractor -> only the convolutional base will be used on top of which a new classifier will be trained.
	After that it will be fine tuned -> some of the convolutional layers will also be trained
	:param train_ds: tf.DataframeIterator
	:param test_ds: tf.DataframeIterator
	:param validate_ds: tf.DataframeIterator
	:return: tf.model
	"""
	inception_model = tf.keras.applications.InceptionV3(include_top=False, input_shape=(256, 256, 3), pooling='avg')
	# freezing the convolutional base
	inception_model.trainable = False
	# define one more fully connected layer
	fcl_layer = tf.keras.layers.Dense(512, activation='relu')
	# define classification layer
	prediction_layer = tf.keras.layers.Dense(8, activation='softmax')
	# define preprocessing layer which rescales pixels to [-1,1]
	preprocess_data = tf.keras.applications.inception_v3.preprocess_input

	# build a model
	# input layer
	inputs = tf.keras.Input(shape=(256, 256, 3))
	# layer with inception preprocessing
	x = preprocess_data(inputs)
	# inception convolutional base
	x = inception_model(x)
	# fcl layer
	x = fcl_layer(x)
	# add dropout
	x = tf.keras.layers.Dropout(0.3)(x)
	# classification, softmax, layer
	outputs = prediction_layer(x)
	# put all of that together
	tl_model = tf.keras.Model(inputs, outputs)

	tl_model.compile(optimizer='adam',
					loss='categorical_crossentropy',
					metrics=['accuracy'])

	history = tl_model.fit(train_ds, epochs=10, validation_data=validate_ds)

	print(tl_model.evaluate(test_ds))

	# fine tune it
	inception_model.trainable = True

	print("Number of layers:", len(inception_model.layers))  # 312

	# Freeze all the layers before the 212th layer
	for layer in inception_model.layers[:212]:
		layer.trainable = False

	tl_model.compile(optimizer='adam',
					loss='categorical_crossentropy',
					metrics=['accuracy'])

	tl_model.fit(train_ds, epochs=15, initial_epoch=history.epoch[-1], validation_data=validate_ds)

	print(tl_model.evaluate(test_ds))

	return tl_model


def load_model(model_name):
	"""
	Load a tf.model from directory
	:param model_name: string, name of the directory
	:return: tf.model
	"""
	return tf.keras.models.load_model(model_name)