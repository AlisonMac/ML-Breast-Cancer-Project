#importing the necessary libraries
import tensorflow as tf
from tensorflow.keras import layers
from skimage.io import imshow
from pathlib import Path
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
from keras.callbacks import Callback
import cv2
import pydicom
import traceback
import cv2
import warnings
import os


print(os.listdir())
dataset_root = Path('d:/')
list(dataset_root.iterdir())
os.chdir(dataset_root)

df = pd.read_csv('dataset.csv', header=None, names=['image_id_new'])
df.head()

#A function to generate the dataframe for a csv file
def generate_df(dataset_root, csv_name):
    df = pd.read_csv(dataset_root/csv_name)
    
    # Check if the 'xmin' column has a valid number
    df['class'] = np.where(pd.isna(df['xmin']), 'negative', 'positive')
    
    # Convert the 'image_id_new' column to string
    df['image_id_new'] = df['image_id_new'].astype(str)

    return df

# Generate a single dataframe
df = generate_df(dataset_root, 'dataset.csv')

# Split the dataframe into train and validation sets
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# Custom function to generate image file paths
def custom_filepath_from_dataframe(row):
    study_id_new = "{:02}".format(row['study_id_new']) 
    image_id_new = row['image_id_new']
    return f"images_png/{study_id_new}/{image_id_new}.png"


# Add a new column 'image_path' to the dataframe with the generated file paths
df['image_path'] = df.apply(custom_filepath_from_dataframe, axis=1)

print(df['image_path'].head())

# Split the dataframe into train and validation sets
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# Dividing the image data generated into train set and validation set
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1. / 255)

# Creating training set
train_gen = datagen.flow_from_dataframe(train_df,
                                        directory=None,
                                        x_col='image_path',
                                        target_size=(224, 224),
                                        class_mode='binary')

# Creating validation set
valid_gen = datagen.flow_from_dataframe(valid_df,
                                        directory=None,
                                        x_col='image_path',
                                        target_size=(224, 224),
                                        class_mode='binary')

#Downloading the densenet model pretrained on the imagenet dataset
densenet = tf.keras.applications.DenseNet169(weights='imagenet', include_top = False, input_shape=(224, 224, 3))

#Freezing the weights of the pretrained model
densenet.trainable = False

#Adding the Flatten layer and the sigmoid classification layer to the pretrained densenet model
model = tf.keras.models.Sequential([
    densenet,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#Compiling the model using adam optimizer
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

class Metrics(Callback):
    def __init__(self, validation_generator):
        super(Metrics, self).__init__()
        self.validation_generator = validation_generator

    def on_train_begin(self, logs={}):
        self.acc_per_epoch = []
        self.val_acc_per_epoch = []
        self.auc_per_epoch = []
        self.sensitivity_per_epoch = []
        self.recall_per_epoch = []

    def on_epoch_end(self, epoch, logs={}):
        validation_steps = len(self.validation_generator)
        y_true = []
        y_pred = []

        # Iterate through the validation data
        for step in range(validation_steps):
            x_val, y_val = next(self.validation_generator)
            y_true.extend(y_val)
            y_pred.extend(self.model.predict(x_val))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Accuracy
        self.acc_per_epoch.append(logs.get('accuracy'))
        self.val_acc_per_epoch.append(logs.get('val_accuracy'))

        # AUC
        auc = roc_auc_score(y_true, y_pred)
        self.auc_per_epoch.append(auc)

        # Sensitivity and Recall
        y_pred_class = (y_pred > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
        sensitivity = tp / (tp + fn)
        recall = recall_score(y_true, y_pred_class)
        
        self.sensitivity_per_epoch.append(sensitivity)
        self.recall_per_epoch.append(recall)

        return

metrics = Metrics(validation_generator=valid_gen)

try:
    history = model.fit_generator(train_gen, epochs=50, validation_data=valid_gen, use_multiprocessing=False, callbacks=[metrics])
except:
    traceback.print_exc()

# Plotting
plt.figure(figsize=(15, 10))
# Accuracy per epoch
plt.subplot(2, 2, 1)
plt.plot(metrics.acc_per_epoch)
plt.plot(metrics.val_acc_per_epoch)
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# AUC per epoch
plt.subplot(2, 2, 2)
plt.plot(metrics.auc_per_epoch)
plt.title('AUC per Epoch')
plt.xlabel('Epoch')
plt.ylabel('AUC')

# Sensitivity per epoch
plt.subplot(2, 2, 3)
plt.plot(metrics.sensitivity_per_epoch)
plt.title('Sensitivity per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Sensitivity')

# Recall per epoch
plt.subplot(2, 2, 4)
plt.plot(metrics.recall_per_epoch)
plt.title('Recall per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Recall')

plt.show()

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
model_new = load_model('model.h5')
model_new.summary()

#with thanks to @hawk453 on Github for their MURA notebook code

