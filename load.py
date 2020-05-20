from scipy.io import loadmat
import numpy as np

def get_data(train_size=7000):
    data = loadmat("./points.mat")
    data = data['xx']
    np.random.shuffle(data)
    train_set = data[:train_size]
    test_set = data[train_size:]
    return train_set,test_set
