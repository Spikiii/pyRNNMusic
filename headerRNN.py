import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
#Following a LSTM Text Generation tutorial from <https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/>

#settings
filename = ""

#defs
raw_text = open(filename).read()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

print(chars)
print(char_to_int)