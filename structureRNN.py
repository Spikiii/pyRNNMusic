import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from genMusic import textSplit, bodyToInt

#Settings
filename = 'Data/abc.txt'
weights_filename = "Checkpoints/structure_0.5828.hdf5"
seq_length = 20 #Length of training sequences to feed into the network
creativity = .5

#Defs
raw_text = ""
text = []
for i in textSplit(filename):
    text.append(i[1])
for i in text:
    for j in bodyToInt(i)[1]:
        raw_text += j
    raw_text += "\n"
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)
dataX = []
dataY = []

#Breaking the text into patterns to feed into network
for i in range(0, n_chars - seq_length):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)

#print("Total Characters: ", n_chars)
#print("Total Vocab: ", n_vocab)
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

def train(e):
    """Trains the network"""
    #Creating checkpoint system
    model.load_weights(weights_filename)
    filepath="Checkpoints/structure_{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
    callbacks_list = [checkpoint]

    # Do the thing!
    model.fit(X, y, epochs=e, batch_size=128, callbacks=callbacks_list)

def generate(leng, log = True):
    """Generates text"""
    #Load the network weights
    model.load_weights(weights_filename)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    #Pick a random seed
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    pattern_text = ""
    for i in range(len(pattern)):
        pattern_text += int_to_char[pattern[i]]
    output = ""

    #Generate characters
    i = 0
    while(i <= leng): #or output[len(output) - 1] != "\n"):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
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
        print(pattern_text + output)
    else:
        return pattern_text + output

#train(20)
generate(500)
#i = input("|||||")
#while(i != "x"):
#    generate(500)
#    i = input("|||||")