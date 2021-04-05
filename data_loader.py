import tensorflow as tf
import os, random
import numpy as np
from keras.preprocessing import image
from PIL import Image, ImageEnhance
import threading
import pandas as pd
from keras.applications import imagenet_utils


class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def preprocess_fn(img, px=(1024, 512)):
    img = img.resize(px, Image.LANCZOS)
    enhance = ImageEnhance.Brightness(img)
    factor = np.random.uniform(low=0.9, high=1.1, size=None)
    img = enhance.enhance(factor)
    img = np.asarray(img)
    img = image.random_rotation(img, 5, row_axis=0, col_axis=1, channel_axis=2)
    return img/255.


@threadsafe_generator
def data_gen(img_folder, df, batch_size, 
                preprocess=preprocess_fn,
                testing=False):
    c = 0
    if testing:
        n = sorted(os.listdir(img_folder))
    else:
        n = os.listdir(img_folder)  
        random.shuffle(n)

    while True:
        img_out = np.zeros((batch_size, 512, 1024, 3)).astype("float")
        label = np.zeros((batch_size, 1)).astype("int")

        for i in range(c, c + batch_size):
            img_name = img_folder + '/'+ n[i]
            if  df.loc[df['ID'] == str(n[i])].size:

                img = image.load_img(
                    img_name
                )
                if preprocess:
                    img = preprocess(img)
                else:
                    img = img.resize((1024, 512), Image.LANCZOS)
                    img = np.asarray(img)
                    img = img/255.     
                # print(n[i])
                index_img = df.loc[df['ID'] == str(n[i])].index[0]

                label[i-c] =  df['Diagnosis'][index_img]              


                img_out[i-c] = img

        c += batch_size
        if c + batch_size >= len(os.listdir(img_folder)):
            c = 0
            random.shuffle(n)
        yield img_out, label
                     
