import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from musicMethods import textSplit
from titleRNN import generate as titleGen
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#Following a LSTM Text Generation tutorial from <https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/>

#Settings
filename = 'Data/abc.txt'
weights_filename = "Checkpoints/header_4.2362.hdf5"
seq_length = 20 #Length of training sequences to feed into the network
creativity = 0.8

#Defs
split_text = textSplit(filename)
raw_text = ""
for i in split_text:
    #header = i[0].split("\n")
    #for j in header:
    #    if(j[:2] == "T:"):
    #        print(j[2:])
    raw_text += i[0]

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)
#print("Total Characters: ", n_chars)
#print("Total Vocab: ", n_vocab)

dataX = []
dataY = []

#Breaking the text into patterns to feed into network
for i in range(0, n_chars - seq_length):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
#print("Total Patterns: ", n_patterns)

X = np.reshape(dataX, (n_patterns, seq_length, 1)) #Reshaping the data for Keras
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

def train(e, load = True):
    """Trains the network"""
    #Creating checkpoint system
    if(load):
        model.load_weights(weights_filename)
    filepath="Checkpoints/header_{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
    callbacks_list = [checkpoint]

    #Do the thing!
    model.fit(X, y, epochs = e, batch_size = 1024, callbacks = callbacks_list)

def generate(leng, log = True):
    """Generates text"""
    #Generates seed
    title_raw = titleGen(200, False).split("\n")
    i = np.random.choice(title_raw[:len(title_raw) - 1])
    if (len("X:1\nT: " + i + "\n") < seq_length):
        for j in title_raw:
            i = j
            if (len("X:1\nT: " + i + "\n") >= seq_length):
                break
    seed_raw = "X:1\nT: " + i + "\n"
    seed = "X:1\nT: " + i + "\n"

    #Filters seed
    if(len(seed) > seq_length):
        seed = seed[len(seed) - seq_length:]

    #Load the network weights
    model.load_weights(weights_filename)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    pattern = [char_to_int[char] for char in seed]
    output = ""

    #Generate characters
    i = 0
    while((i <= leng or output[len(output) - 1] != "\n") and (len(output) == 0 or output[len(output) - 1] != "~")):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose = 0)
        m = max(prediction[0])
        choices = []
        for j in prediction[0]:
            if(j / m >= creativity):
                choices.append(j)
        index = prediction[0].tolist().index(np.random.choice(choices))
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        output += result
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        i += 1
    if(log):
        print(seed_raw + output[:len(output) - 2])
    else:
        return seed_raw + output[:len(output) - 2]

#train(20, False)
#generate(500)