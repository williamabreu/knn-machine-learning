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
    
def KNN_run(dataset_path, split_rate, k):
    data = load_dataset(dataset_path)
    training_data, testing_data = split_dataset(data, split_rate)
    
    classnames = sorted(data.keys())
    confusion_matrix = {predicted: {actual: 0 for actual in classnames} for predicted in classnames}

    for classname in testing_data:
        for testing_instance in testing_data[classname]:
            neighbors = neighborhood(testing_instance, training_data, k)
            prediction = predict(neighbors)
            confusion_matrix[prediction][classname] += 1
            if prediction == classname:
                print('Previsto: {:^16}   |   Real: {:^16} -- CORRETO'.format(prediction, classname))
            else:
                print('Previsto: {:^16}   |   Real: {:^16} -- INCORRETO'.format(prediction, classname))
    
    print()

    for key in confusion_matrix:
        print(key, end=': ')
        print(confusion_matrix[key])
    
if __name__ == '__main__':
    from sys import argv
    dataset_path, split_rate, k = argv[1], float(argv[2]), int(argv[3])
    KNN_run(dataset_path, split_rate, k)