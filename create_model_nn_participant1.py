from numpy import loadtxt
from numpy import savetxt
import numpy
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,AveragePooling1D
from numpy import mean
from tensorflow.random import set_seed
from keras.constraints import min_max_norm
from keras.regularizers import L2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import RobustScaler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import tensorflow

def reshape_function(row):
    return row.reshape(5, 4).transpose().reshape(1, 20)

# setting the seed
seed(1)
set_seed(1)

index1 = 92

# create the robust scaler for the data
rScaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(20, 100-20), unit_variance=True)

# load the data from the csv file
orig_epochs = loadtxt('final_input_data_with_labels.csv', delimiter=',', skiprows=1)
print(orig_epochs.shape)

# scale the data with robust scaler
my_epochs = rScaler.fit_transform(orig_epochs[0:91, 0:3969])  # noise
print(my_epochs.shape)

summed_epochs = my_epochs
print(summed_epochs.shape)

summed_epochs = numpy.append(summed_epochs, orig_epochs[:, -1].reshape(len(orig_epochs[:, -1]), 1), axis=1)
print(orig_epochs[:, -1])

# shuffle the training data
numpy.random.seed(2) 
numpy.random.shuffle(summed_epochs)
print(summed_epochs.shape)

# split the training data between training and validation
tensorflow.compat.v1.reset_default_graph()
X_train_tmp, X_test_tmp, Y_train_tmp, Y_test_tmp = train_test_split(summed_epochs[0:index1, :], orig_epochs[0:index1, -1], random_state=1, test_size=0.3, shuffle = False)
print(X_train_tmp.shape)
print(X_test_tmp.shape)

# augment train data
X_total = X_train_tmp
print(X_total.shape)
print(X_total[:, -1].astype(int).ravel()) 

# data balancing for train data
sm = SMOTE(random_state = 2)
X_train_keep, Y_train_keep = sm.fit_resample(X_total, X_total[:, -1].astype(int).ravel())
print("After OverSampling, counts of label '2': {}".format(sum(Y_train_keep == 2)))
print("After OverSampling, counts of label '1': {}".format(sum(Y_train_keep == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_keep == 0)))
print(X_train_keep.shape)

train_data = numpy.append(X_train_keep, Y_train_keep.reshape(len(Y_train_keep), 1), axis=1)
numpy.random.shuffle(train_data)

#=======================================
 
# Data Pre-processing - scale data using robust scaler

Y_train = train_data[:, -1]
Y_test = X_test_tmp[:, -1]

input = train_data[:, 0:3969]
testinput = X_test_tmp[:, 0:3969]

#=====================================

# Model configuration

input = input.reshape(len(input), 63, 63)
input = input.transpose(0, 2, 1)
print (input.shape)

testinput = testinput.reshape(len(testinput), 63, 63)
testinput = testinput.transpose(0, 2, 1)
print (testinput.shape)

# Create the model
model=Sequential()
model.add(Conv1D(filters=20, kernel_size=6, kernel_regularizer=L2(0.001), bias_regularizer=L2(0.001), activity_regularizer = L2(0.001), kernel_constraint=min_max_norm(min_value=-0.4, max_value=0.4), padding='valid', activation='relu', strides=1, input_shape=(63, 63)))
model.add(Conv1D(filters=20, kernel_size=5, kernel_regularizer=L2(0.001), bias_regularizer=L2(0.001), activity_regularizer = L2(0.001), kernel_constraint=min_max_norm(min_value=-0.4, max_value=0.4), padding='valid', activation='relu', strides=1))
model.add((AveragePooling1D()))
model.add(Conv1D(filters=20, kernel_size=4, kernel_regularizer=L2(0.001), bias_regularizer=L2(0.001), activity_regularizer = L2(0.001), kernel_constraint=min_max_norm(min_value=-0.4, max_value=0.4), padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(20, kernel_regularizer=L2(0.001), bias_regularizer=L2(0.001), activity_regularizer = L2(0.001), kernel_constraint=min_max_norm(min_value=-0.4, max_value=0.4), activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

# Compile the model  
adam = Adam(learning_rate=0.00008)
model.compile(loss=sparse_categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

hist = model.fit(input, Y_train, batch_size=16, epochs=50, verbose=1, validation_data=(testinput, Y_test), steps_per_epoch=None, callbacks=[es, mc])

# evaluate the model
predict_y = model.predict(testinput)
Y_hat_classes=numpy.argmax(predict_y,axis=-1)

matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)


# plot training and validation history
pyplot.plot(hist.history['loss'], label='tr_loss')
pyplot.plot(hist.history['val_loss'], label='val_loss')
pyplot.plot(hist.history['accuracy'], label='tr_accuracy')
pyplot.plot(hist.history['val_accuracy'], label='val_accuracy')
pyplot.legend()
pyplot.xlabel("No of iterations")
pyplot.ylabel("Accuracy and loss")
pyplot.show()

#==================================

model.save("model_conv1d.h5")

# load the best model
saved_model = load_model('best_model.h5')
# evaluate the model
_, train_acc = saved_model.evaluate(input, Y_train, verbose=1)
_, test_acc = saved_model.evaluate(testinput, Y_test, verbose=1)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# evaluate the model
predict_y = saved_model.predict(testinput)
Y_hat_classes=numpy.argmax(predict_y,axis=-1)

matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)


#==================================

