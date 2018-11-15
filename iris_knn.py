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

def euclidean_distance(dataline1, dataline2):
    return sum([(x - y)**2 for x, y in zip(dataline1, dataline2)])

def neighborhood(testing_instance, training_data, k):
    distances = []

    for classname in training_data:
        for training_instance in training_data[classname]:
            distance = euclidean_distance(testing_instance, training_instance)
            distances.append({'distance': distance, 'classname': classname})
    
    distances.sort(key=lambda x: x['distance'])
    return distances[:k]

def predict(neighbors):
    classnames = [x['classname'] for x in neighbors]
    names_count = [{'classname': name, 'count': classnames.count(name)} for name in set(classnames)]
    return max(names_count, key=lambda x: x['count'])['classname']

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
    
def main():
    split_rate = 0.67
    data = load_dataset('dataset/iris_classic.csv')
    training_data, testSet = split_dataset(data, split_rate)
    
    print('Train set: ' + repr(len(training_data)))
    print('Test set: ' + repr(len(testSet)))
    
    # generate predictions
    predictions=[]
    k = 3
    for testline in testSet.values():
        neighbors = neighborhood(list(training_data.values()), testline, k)
        result = predict(neighbors)
        predictions.append(result)
        print(('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1])))
    accuracy = getAccuracy(testSet, predictions)
    print(('Accuracy: ' + repr(accuracy) + '%'))
    
if __name__ == '__main__':
    main()