from keras.preprocessing.image import ImageDataGenerator
from model import model, vgg16
from keras import layers, backend, models, applications, losses, optimizers, metrics, regularizers 
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import tensorflow as tf
from sklearn import metrics as sk_metrics
from collections import Counter
import pandas as pd 
from data_loader import data_gen
import os, sklearn
from activations import GEV


input1 = layers.Input(shape=(512, 1024, 3))
x = applications.InceptionV3(weights='imagenet', include_top=False)(input1)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.6)(x)
out = layers.Dense(1, activation=None, name='label',
                            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-5),
                            dtype='float32'
                            )(x)     
out = GEV()(out)                                                                                                                            
model = models.Model(input1, out)
print(model.summary())



model.compile( 
    loss=losses.binary_crossentropy,
    optimizer=optimizers.Adam(1e-4),
    metrics=[
             metrics.AUC(name='auc'),
             metrics.SensitivityAtSpecificity(0.9, name='sens'),
             metrics.SpecificityAtSensitivity(0.9, name='spec')
             ]    
)
model.load_weights('model.h5')


df = pd.read_csv('../Data/test_labels.csv')
df['ID'] = df['ID'].astype(str) + '.jpg'
df['Diagnosis'] = df['Diagnosis']

true = df['Diagnosis']

test_generator = data_gen(
                    '../Data/test',
                    df,
                    1, testing=True, preprocess=False)
evaluate = model.evaluate(
                    test_generator,
                    steps=172,
                    verbose=1
                    )

test_generator = data_gen(
                    '../Data/test',
                    df,
                    1, testing=True, preprocess=False)
predict = model.predict_generator(
                    test_generator,
                    steps=172,
                    verbose=1
                    )

print(sklearn.metrics.confusion_matrix(true, np.round(predict)))
print(sklearn.metrics.classification_report(true, np.round(predict)))

np.savetxt('model.csv', predict)
np.savetxt('true.csv', true)
