print("Loading\n-----")
from headerRNN import generate as headerGen
print("Header Loaded")
from musicRNNPortable import generate as musicGen
print("Music Loaded\n-----\n")

def genSong(leng):
    output = headerGen(200, False)
    print(output)
    structure = musicGen(leng, False)
    output = ""
    for j in structure:
        output += j
    print(output)

genSong(200)
#while(input("::") != "x"):
#    genSong(200)