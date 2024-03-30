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

