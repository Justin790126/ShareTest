# Regression

* training experiment (tooltips)
  * increase neuron (increase complexity to learn feature in on layer)
  * increase dense layer (increase complexity to learn feature in whole model)
  * optimizer:
    * Adjust learning rate (lr)
    * change optimizer, Adam converage faster than SGD
  * input data preprocess:
    * normalization in scikit learn
   

----

# classification

![test](./resource/classification.png)

[recipe for training neural network](https://karpathy.github.io/2019/04/25/recipe/)

* use early stop callback to find the best learning rate

````
tf.random.set_seed(42)

model9=tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model9.compile(loss="binary_crossentropy",
               optimizer="Adam",
               metrics=["accuracy"])

lrcb=tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4*10**(epoch/20))

his = model9.fit(x_train,y_train, epochs=100,callbacks=[lrcb])
````

````
pd.DataFrame(his.history).plot(figsize=(10,7), xlabel="epochs")
````

````
lrs=1e-4*(10**(tf.range(100)/20))
plt.figure(figsize=(10,7))
plt.semilogx(lrs,his.history["loss"])
plt.xlabel("learning rate")
plt.ylabel("loss")
````

find the learning rate where the loss to be the lowest

![confusionMatrix](./resource/confusionMatrix1.png)

![confusionMatrix2](./resource/confusionMatrix2.png)

----

* ML model baseline is start from the simple one, then add complexity to beat baseline
* padding = "valid", means output tensor would shrink, otherwise output be the same shape
* batch_size = 32, is experience size
* radom.set_seed(42), is experience

> When a model's **validation loss starts to increase**, it's likely that the model is overfitting the training dataset

### induce overfitting:
* increase number of conv layers
* increase number of filters
* add another dense layer to the output of our flattened layer

### reduce overfitting:
* add data augmentation
* add regularization layers (such as MaxPool2D)
  * MaxPool2D: use the rolling window to preserve the most significant features
* add more data


### Shuffle augemented data v.s. augemented data
>  shuffle generate randomness on data to make model learn more generally to unseen data
> make model to learn equally on different kinds of data

### optimization direction in CNN

* deeper layers
* wider neurons
* best learning rate
* train more epochs
* more data
* transfer learning


### Step to create data set for multiclass classification

* ImageDataGenerator
  * can use data augmentation here
* flow from directory

### Callbacks
* TensorBoard callback
* ModelCheckpoint callback
* EarlyStopping callback
  * aim to find best epochs

### Funtional API

It is better to explore tensors layer by layer

> If it is a layer, put input x outside the bracket
> ex. GlobalAveragePooling2D()(x)

> If it is a model, put input x inside the bracket
> ex. base_model(x)

````
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)

base_model.trainable = False

inputs = tf.keras.layers.Input(shape=(224,224,3), name="input_layer")

# x = tf.keras.layers.experimental.preprocessing.Rescale(1./255)(inputs)

x = base_model(inputs)
print(f"Shape after passing inputs through base model:{x.shape}")

x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_kayrt")(x)
print(f"Shape after GlobalAveragePooling2D:{x.shape}")

outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_layer")(x)

model0 = tf.keras.Model(inputs, outputs)

model0.compile(loss="categorical_crossentropy",
               optimizer=tf.keras.optimizers.Adam(),
               metrics=["accuracy"])

his0 = model0.fit(train_data_10_percent,
                  epochs=5,
                  steps_per_epoch=len(train_data_10_percent),
                  validation_data=test_data,
                  validation_steps=int(0.25*len(test_data)),
                  callbacks=[create_tensorboard_callback("transfer_learning",
                                                         "10_percent_feature_extraction")]
                  )
````

### preprocessing layer can do data augementation in GPU

````
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

data_aug = keras.Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2),
    preprocessing.Rescaling(1./255)
], name="data_aug")
````

````
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
#define base model
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

#define inputs

inputs = Input(shape=(224,224,3), name="inputs")

x = data_aug(inputs)

x = base_model(x, training=False)

x = GlobalAveragePooling2D()(x)

outputs = Dense(10, activation="softmax", name="outputs")(x)

#define outputs
model2 = tf.keras.Model(inputs=inputs, outputs=outputs)

model2.compile(loss="categorical_crossentropy",
               optimizer=tf.keras.optimizers.Adam(),
               metrics=["accuracy"])

his2=model2.fit(
    train_data_10_percent,
    epochs=5,
    steps_per_epoch=len(train_data_10_percent),
    validation_data=test_data,
    validation_steps=len(test_data),
)
````

### In fine tunning, we better lower the learning rate to 10x

* unfreeze the last 5 layers
````
base_model.trainable = True
for layer in base_model.layers[:-5]:
  layer.trainable = False
````

* In multi-class classification, shuffle=False means keep the order of test_data to compare with the output prediction labels

> and we can unbatch to see the test_data ylabels
````
import tensorflow as tf
IMG_SIZE=(224,224)
train_data_all_10_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                       label_mode="categorical",
                                                                       image_size=IMG_SIZE)
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE,
                                                                shuffle=False)
````


````
data.shape = (2, 101)
data.argmax(axis=1) will perform maximum on each 101 dimension data
````

### Evluation a mult-class classification

> finally, write a random functions to predict and compare to the ground truth

* inspect wrong preidctions with the highest prediction probability
* Root cause:
  * Data issues(wrong labels, but model is true because the wrong label had fitted in)
  * confusing classes(get better/more diverse data)


## Finding the most wrong predictions
1. Get all of the image file paths in the test dataset using list_files() method
2. Create dataframe of image filepaths, ground truth labels, predicted classes from model, max prediction probability
3. Use DataFrame to find all wrong predictions
4. Sort the dataframe based on wrong preidction (highest prediction probability at the top)
5. Visualize the images with highest prediction probabilities but have the wrong prediction


* iterate data frames

````
for i, row in enumerate(top_100_wrong[start_index: start_index+images_to_view].itertuples()):
  print(row)
````


* when predict a single image on model, we should expand_dim to let batch size to be 1


----
# NLP

Tokenization : convert string to integer
Embedding : convert string to float array, which had richer representation


start from scikit machine learning map


### RNN

| Name        | When to use           | Code  |
| ------------- |:-------------:| -----:|
| LSTM        | default for sequence problems      | tf.keras.layers.LSTM |
| GRU        | simliar to LSTM (can be default)      |   tf.keras.layers.GRU |
| Bidirectional LSTM      | Good for sequences which may benifit from passing forwards/backwards(translation, or longer passages of text)      |    tf.keras.layers.Bidirectional |





````
for (int i = 0; i < 4; i++) {
  #pragma omp parallel
  {
    #pragma omp parallel for nowait
    for (int j = 0; j < 100; j++) {

    }
    #pragma omp barrier
  }
}
````