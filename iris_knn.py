# Example of kNN implemented from Scratch in Python

import random
import math
import operator

def load_dataset(file_path):
    classes = None

    with open(file_path) as dataset:
        classes = {}
        header = dataset.readline()[:-1].split(',')
        data = [line[:-1].split(',') for line in dataset]
        data.sort(key=lambda x: x[-1])
        for i in range(len(data)):
            classname = data[i][-1]
            dataline = list(map(float, data[i][:-1]))
            if classname in classes:
                classes[classname].append(dataline)
            else:
                classes[classname] = [dataline]

    return classes

def split_dataset(data, rate):
    training_data = {}
    testing_data = {}
    
    for classname in data:
        dataline = data[classname]
        partition = int(len(dataline) * rate)
        training_data[classname] = dataline[:partition]
        testing_data[classname] = dataline[partition:]

    return training_data, testing_data

def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def neighborhood(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclidean_distance(testInstance, trainingSet[x], length)
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
    header, classes, data = load_dataset('dataset/iris_classic.csv')
    ##########################
    print(classes)
    print(header)
    for i in data:
        print(i)
    input()
    ##########################
    trainingSet, testSet = split_dataset(data, split_rate)
    
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    
    # generate predictions
    predictions=[]
    k = 3
    for x in range(len(testSet)):
        neighbors = neighborhood(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print(('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1])))
    accuracy = getAccuracy(testSet, predictions)
    print(('Accuracy: ' + repr(accuracy) + '%'))
    
if __name__ == '__main__':
    main()