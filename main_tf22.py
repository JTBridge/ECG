from keras.preprocessing.image import ImageDataGenerator
from model import model, vgg16
from keras import layers, backend, models, applications, losses, optimizers, metrics, regularizers 
from keras.preprocessing import image
from keras import backend as K
import numpy as np
from sklearn import metrics as sk_metrics
from collections import Counter
import pandas as pd 
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_loader import data_gen
from activations import GEV

df = pd.read_csv('../Data/train_labels.csv')
df['ID'] = df['ID'].astype(str) + '.jpg'
df['Diagnosis'] = df['Diagnosis'].astype(str) 

batch_size=10
train_generator = data_gen(
                    '../Data/train/',
                    df,
                    batch_size)

val_generator = data_gen(
                    '../Data/val/',
                    df,
                    batch_size,
                    preprocess=False, testing=True)
 

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


STEP_SIZE_TRAIN=800//batch_size
STEP_SIZE_VALID=200//batch_size

reduceLROnPlat = ReduceLROnPlateau(
    monitor="val_auc", factor=0.67, patience=3, verbose=1, mode="max", epsilon=0.0001)
earlyStopping = EarlyStopping(
    monitor='val_auc', patience=15, verbose=0, mode='max')
mcp_save = ModelCheckpoint(
    'model.h5', save_best_only=True, monitor='val_auc', mode='max', verbose=1)

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_generator,
                    validation_steps=STEP_SIZE_VALID,
                    class_weight={0:1, 1:(800-201)/201},
                    epochs=250,
                    callbacks=[reduceLROnPlat,
                               earlyStopping,
                               mcp_save],                    
                )
