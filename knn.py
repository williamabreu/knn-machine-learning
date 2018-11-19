# Implementação do algoritmo KNN
# 
# Funciona a partir de uma base de dados em que parte 
# dela será usada para treinamento e o restante dos
# dados serão usados para testar a taxa de reconhecimento
# do sistema implementado.
# 
# A implementação está de maneira independente da instância,
# assim qualquer problema KNN pode ser resolvido por ela,
# desde que seja obedecida a formatação do arquivo CSV.


def load_dataset(file_path: str) -> dict:
    """
    Carrega o arquivo CSV do dataset em JSON para o sistema
    """
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


def split_dataset(data: dict, rate: float) -> (dict, dict):
    """
    Divide o dataset JSON em dois subconjuntos disjuntos, um
    para treinar e o outro para testar o sistema
    """
    training_data = {}
    testing_data = {}
    
    for classname in data:
        dataline = data[classname]
        partition = int(len(dataline) * rate)
        training_data[classname] = dataline[:partition]
        testing_data[classname] = dataline[partition:]

    return training_data, testing_data


def euclidean_distance(dataline1: list, dataline2: list) -> float:
    """
    Calcula a distância euclideana entre duas instâncias do dataset
    """
    return sum([(x - y)**2 for x, y in zip(dataline1, dataline2)])


def neighborhood(testing_instance: list, training_data: dict, k: int) -> list:
    """
    Calcula os k vizinhos de uma instância do dataset
    """
    distances = []

    for classname in training_data:
        for training_instance in training_data[classname]:
            distance = euclidean_distance(testing_instance, training_instance)
            distances.append({'distance': distance, 'classname': classname})
    
    distances.sort(key=lambda x: x['distance'])
    return distances[:k]


def predict(neighbors: list) -> str:
    """
    Calcula qual classe tem maior incidência nos vizinhos por
    maioria simples
    """
    classnames = [x['classname'] for x in neighbors]
    names_count = [{'classname': name, 'count': classnames.count(name)} for name in set(classnames)]
    return max(names_count, key=lambda x: x['count'])['classname']
    

def KNN_run(dataset_path: str, split_rate: float, k: int) -> None:
    """
    Chamada principal do algoritmo KNN
    """
    data = load_dataset(dataset_path)
    training_data, testing_data = split_dataset(data, split_rate)
    
    classnames = sorted(data.keys())
    confusion_matrix = {predicted: {actual: 0 for actual in classnames} for predicted in classnames}
    correct_count = 0
    incorrect_count = 0
    
    # Codificação de cores do terminal:
    #   \x1B[0m  -- padrão
    #   \x1B[1m  -- negrito
    #   \x1B[31m -- texto vermelho
    #   \x1B[41m -- fundo vermelho
    #   \x1B[32m -- texto verde
    #   \x1B[42m -- fundo verde
    strformat = 'Previsto -> \x1B[1m{0}{1:>20}\x1B[0m  x  \x1B[1m{0}{2:<20}\x1B[0m <- Real    {3}{4:^11}\x1B[0m'

    for classname in testing_data:
        for testing_instance in testing_data[classname]:
            neighbors = neighborhood(testing_instance, training_data, k)
            prediction = predict(neighbors)
            confusion_matrix[prediction][classname] += 1
            
            # Reconhecimento da instância:
            if prediction == classname:
                print(strformat.format('\x1B[32m', prediction, classname, '\x1B[42m', 'CORRETO'))
                correct_count += 1
            else:
                print(strformat.format('\x1B[31m', prediction, classname, '\x1B[41m', 'INCORRETO'))
                incorrect_count += 1

    # Matriz de confusão:
    print()
    print('Matriz de confusão:')
    print('-' * 21 * (len(classnames) + 1))
    print(''.center(20), end=' ')
    for key in classnames:
        print('\x1B[1m{:^20}\x1B[0m'.format(key), end=' ')
    print()
    for key1 in classnames:
        print('\x1B[1m{:^20}\x1B[0m'.format(key1), end=' ')
        for key2 in classnames:
            if key1 == key2:
                print('\x1B[32m{:^20}\x1B[0m'.format(confusion_matrix[key1][key2]), end=' ')
            else:
                print('\x1B[31m{:^20}\x1B[0m'.format(confusion_matrix[key1][key2]), end=' ')
        print()
    linesize = 21 * (len(classnames) + 1)
    print('-' * linesize)
    print()

    # Acurácia:
    accuracy = 100 * correct_count / (correct_count + incorrect_count)
    print('Acurácia: {:.4f} %'.format(accuracy).center(linesize))
    print()  

    
if __name__ == '__main__':
    # Uso pelo terminal:
    # python3 knn.py <str: dataset.csv> <float: taxa_de_divisão> <int: k>

    from sys import argv
    dataset_path, split_rate, k = argv[1], float(argv[2]), int(argv[3])
    KNN_run(dataset_path, split_rate, k)