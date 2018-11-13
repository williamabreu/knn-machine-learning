# Example of kNN implemented from Scratch in Python

import random
import math
import operator

def load_dataset(file_path):
    header = data = None

    with open(file_path) as dataset:
        header = dataset.readline()[:-1].split(',')
        data = [line[:-1].split(',') for line in dataset]
        for i in range(len(data)):
            data[i] = list(map(float, data[i][:-1])) + [data[i][-1]]

    return header, data

def split_dataset(data, split):
    trainingSet=[]
    testSet=[]
    for x in range(len(data)-1):
        if random.random() < split:
            trainingSet.append(data[x])
        else:
            testSet.append(data[x])

    return trainingSet, testSet

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(iter(classVotes.items()), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
    
def main():
    split_rate = 0.67
    header, data = load_dataset('dataset/iris_classic.csv')
    trainingSet, testSet = split_dataset(data, split_rate)
    
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    
    # generate predictions
    predictions=[]
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print(('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1])))
    accuracy = getAccuracy(testSet, predictions)
    print(('Accuracy: ' + repr(accuracy) + '%'))
    
main()