from os import listdir, path
from os.path import isfile, join

def get_files():
    return [f for f in listdir(path.abspath("")) if isfile(join(path.abspath(""), f))]