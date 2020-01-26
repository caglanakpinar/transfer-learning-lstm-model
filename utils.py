from os import listdir, path
from os.path import isfile, join

def get_files():
    return [f for f in listdir(path.abspath("")) if isfile(join(path.abspath(""), f))]

reshape_3 = lambda x: x.reshape((x.shape[0], x.shape[1], 1))
reshape_2 = lambda x: x.reshape((x.shape[0], 1))