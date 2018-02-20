import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys
#Following a LSTM Text Generation tutorial from <https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/>

#Settings
filename = 'Data/navySeal.txt'
weights_filename = "Checkpoints/weights_09_0.8179.hdf5"
seq_length = 25 #Length of training sequences to feed into the network

#Defs
raw_text = open(filename).read()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

dataX = []
dataY = []

#Breaking the text into patterns to feed into network
for i in range(0, n_chars - seq_length):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

X = numpy.reshape(dataX, (n_patterns, seq_length, 1)) #Reshaping the data for Keras
X = X / float(n_vocab) #Normalizing the data
y = np_utils.to_categorical(dataY) #Something about hot encoding?

#Defining the Model
model = Sequential()
model.add(LSTM(256, input_shape = (X.shape[1], X.shape[2]), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation = "softmax")) #Output layer
model.compile(loss = "categorical_crossentropy", optimizer = "adam")

def train(e):
    """Trains the network"""
    #Creating checkpoint system
    model.load_weights(weights_filename)
    filepath="Checkpoints/weights_{epoch:02d}_{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
    callbacks_list = [checkpoint]

    #Do the thing!
    model.fit(X, y, epochs = e, batch_size = 128, callbacks = callbacks_list)

def generate(leng):
    """Generates text"""
    #Load the network weights
    model.load_weights(weights_filename)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    # pick a random seed
    start = numpy.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]

    output = ""
    # generate characters
    for i in range(leng):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        output += result
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("".join([int_to_char[value] for value in pattern]) + "|" + output)

generate(1000)
