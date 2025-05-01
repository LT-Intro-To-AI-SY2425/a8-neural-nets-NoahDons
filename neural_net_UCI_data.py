from typing import Tuple
from neural import *
from sklearn.model_selection import train_test_split
from neural import NeuralNet

def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    tokens = line.split(",")
    output = [tokens[5]]
    #output = [1 if out == 1 else 0.5 if out == 2 else 1]

    inpt = [float(x) for x in tokens[:5]]
    return (inpt, output)


def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
    return data

increment = 10
i=0
dataset = []
with open("wine-quality-white-and-red.csv", "r") as file:
    lines = file.readlines()

header = True
for line in lines:
    if header:
        header = False
        continue  
    
    parts = line.strip().split(",")
    wine_type = parts[0].lower()        
    features = parts[1:]                

    
    input_vector = [float(x) for x in features]
    

    if wine_type == "red":
        label = [1.0]
    elif wine_type == "white":
        label = [0.0]
    
    if i == increment:
        dataset.append((input_vector, label))
        i=0
    else:
        i+=1
    

normalized = normalize(dataset)

model = NeuralNet(12, 1, 1)
model.train(normalized)
tests = model.test_with_expected(normalized)
correct = 0
incorrect = 0
for test in tests:
    print("Actual:", test[1])
    print("Predicted:", test[2])
    if int(test[2][0]+.5) == int(test[1][0]):
        correct+=1
    else:
        incorrect+=1
print(correct)
print(incorrect)
