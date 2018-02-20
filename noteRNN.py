

#Settings
filename = "Data/abc.txt"

#Defs
raw_file = open(filename)
print(raw_file)
raw_text = []

for line in raw_file:
    raw_text.append(line)

X = []

def textSplit():
    global X
    header = []
    body = []

    headerStart = 0
    bodyStart = 0
    for i in range(len(raw_text)):
        if(raw_text[i][0] == "K"):
            for j in range(headerStart, i + 1):
                header.append(raw_text[j])
            bodyStart = i + 1
        if(raw_text[i] == "\n"):
            for j in range(bodyStart, i):
                body.append(raw_text[j])
            X.append((header, body))
            header = []
            body = []
            headerStart = i

textSplit()
print(X)
