# THEANO_FLAGS=device=gpu,floatX=float32 python train.py
# bug: training length should be larger than batch size
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.models import load_model
import numpy as np
import random
import sys
import copy
import time
import re
import pysynth

### Define some constants
maxlen = 16 # The length of LSTM
epochs = 50

### Read the file and put the music into a list
def splitMusic(file_name = "dataset/data.txt"):
	print("...Reading file %s and count the number of songs..." %(file_name))
	with open(file_name, "r") as f:
		m = f.readlines()
	print ("%d songs found." %(len(m)))
	m = [re.split('[\[\],\s\']*', p)[1:-1] for p in m]
	return m

### Make the dict from index to length&note and reverse
def makeDict(melody):
	print("...Making the dict from index to length&note and reverse...")
	l = set()
	n = set()
	for melo in melody:
		l = l | set(melo[1 : : 2])
		n = n | set(melo[0 : : 2])
	print ("%d type of note duration" %(len(l)))
	print ("%d type of notes" %(len(n)))
	dic_size = len(l) * len(n)
	print ("input dict length: %d" %(dic_size))
	m_to_i = {}
	i_to_m = {}
	count = 0
	for a in l:
		for b in n:
			m_to_i[a + '\t' + b] = count
			i_to_m[count] = a + '\t' + b
			count += 1
	return m_to_i, i_to_m, dic_size

### Form the training sets
def makeTrainset(melody, m_to_i):
	print("...Making the training set...")
	X = []
	y = []
	for i,melo in enumerate(melody):
		ind = [m_to_i[melo[2*i+1] + '\t' + melo[2*i]] for i in range(len(melo) // 2)]
		for i in range(len(ind) - maxlen):
			X.append(ind[i : i + maxlen])
			y.append(ind[i + maxlen])
	dic_size = len(m_to_i)
	X_train = np.zeros((len(X), maxlen, dic_size), dtype=np.bool)
	y_train = np.zeros((len(X), dic_size), dtype=np.bool)
	for i, m in enumerate(X):
	    for t, n in enumerate(m):
	        X_train[i, t, n] = 1
	    y_train[i, y[i]] = 1
	print ("Training set size: X_train", X_train.shape)
	print ("Training set size: y_train", y_train.shape)
	return X_train, y_train

### Make the LSTM-model and training
def trainModel(X_train, y_train, seq_length, dic_size):
	print ("...Making the LSTM-model...")
	model = Sequential()
	model.add(LSTM(512, return_sequences=True, input_shape=(seq_length, dic_size)))
	model.add(Dropout(0.2))
	model.add(LSTM(512, return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(dic_size))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	print ("...Start training...")
	model.fit(X_train, y_train, batch_size=256, nb_epoch=epochs)
	return model

### Predict the output notes with feedin and write it out to music_out.txt
def predict(model, feedin, len, i_to_m):
	predi = []
	m = copy.deepcopy(feedin)
	for i in range(len):
		y = model.predict(np.array([m]))
		note = list(y[0]).index(max(y[0]))
		predi.append(note)
		n = copy.deepcopy(m)
		n[0 : maxlen - 1] = m[1 : ]
		n[-1] = np.zeros(dic_size, dtype=np.bool)
		n[-1][note] = 1
		m = n
	# localtime = time.asctime( time.localtime(time.time()) )
	# print("...Writing file back to %s" %("music_out_" + localtime+ ".txt"))
	# with open("music_out_" + localtime+ ".txt", "w") as f:
	# 	for i in predi:
	# 		f.write(i_to_m[i] + '\t')
	return(predi)

### The main program
melody = splitMusic("dataset/data.txt")
melo_to_index, index_to_melo, dic_size = makeDict(melody)
X_train, y_train = makeTrainset(melody, melo_to_index)
X_train, y_train, X_test, y_test = X_train[0:80000], y_train[0:80000], X_train[-20000:], y_train[-20000:]
model = trainModel(X_train, y_train, maxlen, dic_size)
print ("..Training is finished...")

### Predict the music comp
comp = [predict(model, X_test[i], 200, index_to_melo) for i in range(0,20000,1000)]
wave = [[index_to_melo[i] for i in c] for c in comp]
wave = [[[m.split()[1], float(m.split()[0])] for m in w] for w in wave]
for i,w in enumerate(wave):
	pysynth.make_wav(w, fn = "composed_melody/test/test"+str(i)+".wav")

### Save the model
model.save('trained_model/my_model.h5')
