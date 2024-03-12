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