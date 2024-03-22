#!/bin/python3

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def plot_loss_curves(his):
  loss = his.history["loss"]
  val_loss= his.history["val_loss"]

  acc = his.history["accuracy"]
  val_acc = his.history["val_accuracy"]

  epochs=range(len(his.history["loss"]))

  plt.figure()
  plt.plot(epochs, loss, label="training_loss")
  plt.plot(epochs, val_loss, label="val_loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  plt.figure()
  plt.plot(epochs, acc, label="training_acc")
  plt.plot(epochs, val_acc, label="val_acc")
  plt.title("acc")
  plt.xlabel("epochs")
  plt.legend()

def view_random_image(target_dir, target_class):
  target_folder = target_dir+target_class

  random_image = random.sample(os.listdir(target_folder), 1)
  print(random_image)
  img = mpimg.imread(target_folder+"/"+random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off")
  print(f"shape: {img.shape}")
  return img


