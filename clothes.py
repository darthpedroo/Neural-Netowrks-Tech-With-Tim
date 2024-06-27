#https://www.tensorflow.org/tutorials/keras/classification
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] #test_label returns a value between 0-9 and this are the different categories

print("Size of train_images:", train_images.shape) #The first value is the number of images and the remaining 2 are the width and height of each image

print("Size of test_images:", test_images.shape) #Test images has significantly less images

print("TRAIN IMAGE BIT :", train_images[69])

train_images = train_images/255.0
test_images = test_images/255.0 #Hacemos estos para tener valores entre 0 y 1 en vez de entre 0 y 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"), #relu = Rectify linear unit
    keras.layers.Dense(10, activation="softmax"), #SOFTMAax es la probabilidad de que cada neurona de un valor de posibilidad entre 0 - 100 % converts the raw output scores (logits) of the network into probabilities, making it easier to interpret the results
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5) #epochs = how many times a model will see the same image


prediction = model.predict(test_images)

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(prediction[i])])

plt.show()

#test_loss, test_acc = model.evaluate(test_images, test_labels)

#print("Tested ACC: ", test_acc)

#plt.figure(figsize=(10,10))


#We need to flatten the data, so instead of being 28*28 it is 784 
#ARCHITECTURE OF THE NET
#INPUT = 784 PIXELES
#HIDDEN LAYER: 128 NEURONS, ALLOWS MUCH MORE COMPLEXITY 
#10 neurons, each one represent the category class
#7840 WAVES AND BIAS