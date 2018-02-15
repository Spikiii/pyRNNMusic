import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
#Following a LSTM Text Generation tutorial from <https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/>

#settings
filename = 'Data/abc.txt'
seq_length = 50 #Length of training sequences to feed into the network

#defs
raw_text = open(filename).read()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)

print("Total Chars: " + n_chars)
print("Total Vocab: " + n_vocab)

dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1): #Breaking up the patterns to feed into the network
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)

print("Total Patters: ", n_patterns)

X = numpy.reshape(dataX, (n_patterns, seq_length, 1)) #Reshaping for Keras
X = X / float(n_vocab) #Normaize the data
y = np_utils.to_categorical(dataY) #Something about hot encoding?

model = Sequential()
model.add(LSTM(256, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation = "softmax"))
model.compile(loss = "categorical_crossentropy", optimizer = "adam")

