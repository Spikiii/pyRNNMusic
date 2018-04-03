print("Loading\n-----")
from headerRNN import generate as headerGen
print("Header Loaded")
from structureRNN import generate as structureGen
print("Structure Loaded")
from notesRNN import generate as notesGen
print("Notes Loaded\n-----\n")

def genSong(leng):
    output = ""
    output += headerGen(200, False)
    print(output)
    structure = structureGen(leng, False)
    notes = notesGen(structure, False)
    noteOccs = []  # Lists indexes of Note Occurances
    for i in range(len(structure)):
        if (structure[i] == "~"):
            noteOccs.append(i)
    notesPos = 0
    output = ""
    for j in structure:
        if(j == "~"):
            output += notes[notesPos]
            notesPos += 1
        else:
            output += j
    print(output)

genSong(200)
#while(input("::") != "x"):
#    genSong(200)