import sys
#add path of the project folder on your laptop here 
#sys.path.append('C:/Users/Moi/Desktop/AARN_2')
sys.path.append('YOUR PATH')
import numpy as np
from keras import Sequential
import keras.backend as K
from keras.layers import Dense, Dropout
from Dataset.tools import getDataset
from variation import getVariations
from keras import metrics
import matplotlib.pyplot as plt
#retrieve dataset
x_train, x_test, y_train, y_test = getDataset('../Dataset/features.xlsx')

#retrieve architectures variation
variation = getVariations()

best_accuracy = -1

#retrieve the shape of data
n_input = x_train[0].shape

#loop through the architectures
for v in variation:
	#initialzie the NN
	model = Sequential()
	#set the layers
	for i in range(len(v[-1])):
		if (i == 0): 
			model.add(Dense(v[-1][i],input_shape=n_input, use_bias=True, 
				kernel_initializer='RandomUniform', bias_initializer='zeros', activation=v[0]))
			model.add(Dropout(v[2], noise_shape=None, seed=None))
		else:
			model.add(Dense(v[-1][i], use_bias=True, 
				kernel_initializer='RandomUniform', bias_initializer='zeros', activation=v[0]))
			model.add(Dropout(v[2], noise_shape=None, seed=None))

	#set the output layer
	model.add(Dense(1, use_bias=True, 
		kernel_initializer='RandomUniform', bias_initializer='zeros', activation=v[0]))

	#compile and fit we use 200 epochs cause we have no time to run longer tests
	model.compile(optimizer=v[1] ,loss='mean_squared_error', metrics=['accuracy'])
	history = model.fit(x_train, y_train, batch_size=100, epochs=200, verbose=2,validation_split= 0.5, validation_data = (x_test, y_test))
	accuracy = model.evaluate(np.split(x_test,2)[1], np.split(y_test,2)[1], verbose = 0 )

	#test the model results and save the best model generated
	if (accuracy[1] > best_accuracy):
		best_accuracy = accuracy[1]
		best_model = model
		best_varitaion = v
		best_history = history

print(accuracy)
#print('Accuracy: %.2f' % (best_accuracy*100))

#print the best architecture obtained
print(K.eval(best_model.optimizer.lr))
for i in best_varitaion:
	print(i)

	
#print(best_history.history['acc'])
#print(best_history.history['val_acc'])
#print(best_history.history['loss'])
#print(best_history.history['val_loss'])

# Plot training & validation accuracy values
plt.plot(best_history.history['acc'])
plt.plot(best_history.history['loss'])
plt.plot(best_history.history['val_acc'])
plt.plot(best_history.history['val_loss'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['acc', 'loss', 'valid_acc', 'valid_loss'], loc='upper left')
plt.show()

#save file
best_model.save('best_model.h5')
print("Saved model to disk")



'''
#model loading
loaded_model = load_model('best_model.h5')
accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))

'''
