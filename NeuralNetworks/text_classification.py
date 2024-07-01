import tensorflow as td
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb
num_words = 88000

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=num_words)

word_index = data.get_word_index() 

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])


train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def review_encode(s):
    encoded = [1] #Serves as the <START> TAG

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()]) #Si está, añade la palabra 
        else:
            encoded.append(2) #Si no está, añade un <UKN>

    return encoded

model = keras.models.load_model("model.h5")

with open('test.txt', encoding="utf-8") as f:
    counter = 0
    for line in f.readlines():
        counter+=1
        print("LINE TYPUE: ", type(line))
        nline  = line.replace(',', "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print("Line: \n", line)
        print("Encode: \n", encode)
        print("Predict: \n", predict[0])

    print("COUNTER: ", counter)


"""
model = keras.Sequential()
model.add(keras.layers.Embedding(num_words, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid")) #sigmoid es la que convierte entre los valores entre 0 y 1. AH = 0 , AH = 1 x= 0 y = 0.5

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy"]) #calculate the difference betwenn, ie: 0.2 and 0. we use it cause whe have just 2 values

x_validation_data = train_data[:10000] #validation data
x_train = train_data[10000:]

y_validation_data = train_labels[:10000]
y_train = train_labels[10000:]

                                                        
fitModel = model.fit(x_train, 
					 y_train, 
					 epochs=40, 
					 batch_size=512, #how many reviews we are gonna load at once
					 validation_data=(x_validation_data,y_validation_data),verbose =1)

results = model.evaluate(test_data, test_labels)

print(results)

test_review = test_data[0]
predict = model.predict(np.expand_dims(test_review, axis=0))  # Reshape test_review to be 2D

print("Review: ")
print(decode_review(test_review))

print("Prediction: "+str(predict[0]))
print("Actual: "+ str(test_labels[0]))

print("Results: ")
print(results)

#model.save("model.h5")
"""
