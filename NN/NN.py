import sys
#add path of the project folder on your laptop here 
sys.path.append('C:/Users/Moi/Desktop/AARN_2')
import numpy as np
from keras import Sequential
import keras.backend as K
from keras import optimizers
from keras.layers import Dense,  Embedding, LSTM, GRU,Dropout
from DATASETS.tools import getDataset
from keras.models import load_model
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = getDataset('../Dataset/features.xlsx')


n_input = x_train[1].shape
n_classes = 1

neurons_per_layer=[10,10]
activation = ['tanh']
opt = ['Adam']
learning_rate = 0.1	

model = Sequential()

model.add(Dense(neurons_per_layer[0],input_shape=n_input, use_bias=True, 
	kernel_initializer='RandomUniform', bias_initializer='zeros', activation=activation[0]))

model.add(Dropout(0.1, noise_shape=None, seed=None))

model.add(Dense(neurons_per_layer[1], use_bias=True, 
	kernel_initializer='RandomUniform', bias_initializer='zeros', activation=activation[0]))
model.add(Dropout(0.1, noise_shape=None, seed=None))

model.add(Dense(1, use_bias=True, 
	kernel_initializer ='RandomUniform', bias_initializer='zeros', activation=activation[0]))

a = optimizers.Adam(lr= learning_rate)
model.compile(optimizer=a ,loss='mean_squared_error', metrics=['accuracy'])

best_history = model.fit(x_train, y_train, batch_size=100, epochs=200, verbose=2,validation_split= 0.5, validation_data = (x_test, y_test))

accuracy = model.evaluate(np.split(x_test,2)[1], np.split(y_test,2)[1], verbose = 0 )
#print('Accuracy: %.2f' % (accuracy*100))

print(accuracy)

print(best_history.history.keys())

#print('Accuracy: %.2f' % (accuracy*100))
#save file
model.save('model.h5')
print("Saved model to disk")

#print(best_history.history['acc'])
#print(best_history.history['val_acc'])

# Plot training & validation accuracy values
plt.plot(best_history.history['acc'])
plt.plot(best_history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['acc', 'valid_acc'], loc='upper left')
plt.show()


#print(best_history.history['loss'])
#print(best_history.history['val_loss'])
# Plot training & validation accuracy values
plt.plot(best_history.history['loss'])
plt.plot(best_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['loss','valid_loss'], loc='upper left')
plt.show()
'''
#model loading
loaded_model = load_model('model.h5')
_, accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))
'''