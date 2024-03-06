from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Create the model
model = Sequential()

# Add layers to the model
model.add(InputLayer(batch_input_shape=(None, 48, 48, 1), dtype='float32'))
model.add(Conv2D(128, kernel_size=3, strides=1, padding='valid', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model.add(Dropout(0.4))
model.add(Conv2D(256, kernel_size=3, strides=1, padding='valid', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model.add(Dropout(0.4))
model.add(Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model.add(Dropout(0.4))
model.add(Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(512, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(Dropout(0.3))
model.add(Dense(27, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()