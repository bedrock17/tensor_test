
from PIL import Image
import numpy as np

def loadImageArray(filePath :str):
    img = Image.open(filePath)
    d = img.getdata()
    d = np.array(d)

    lst = []
    for i in d:
        t = 0
        if i[0] > 128:
            t = 1
        lst.append(t)

    return lst

def TestCode():
    x_lst = []
    x_lst.append(loadImageArray("1.png"))
    x_lst.append(loadImageArray("2.png"))

    print(x_lst)