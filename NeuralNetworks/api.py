from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from flask_cors import CORS
import tensorflow as td
from tensorflow import keras
import numpy as np

app = Flask(__name__)
api = Api(app)
CORS(app)

class TextModel:
    def __init__(self):
        self.data = keras.datasets.imdb
        self.__model = keras.models.load_model("model.h5")
        self.word_index = self.data.get_word_index()
        self.word_index = {k: (v + 3) for k, v in self.word_index.items()}
        self.word_index["<PAD>"] = 0
        self.word_index["<START>"] = 1
        self.word_index["<UNK>"] = 2
        self.word_index["<UNUSED>"] = 3
        self.reverse_word_index = {value: key for key, value in self.word_index.items()}

    @property 
    def model(self):
        return self.__model

    def decode_review(self, text):
        return " ".join([self.reverse_word_index.get(i, "?") for i in text])
    
    def review_encode(self, words):
        encoded = [1]  # Serves as the <START> TAG
        for word in words:
            if word.lower() in self.word_index:
                encoded.append(self.word_index[word.lower()])  # If the word exists, add it
            else:
                encoded.append(2)  # If the word doesn't exist, add <UNK>
        return encoded


    def clean_review(self, review):
        print("REVIEW: ", type(review))
        return review.replace(',', "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").replace("/", "").strip().split(" ")

    def predict_review(self, review):
        clean_review = self.clean_review(review)
        encoded_review = self.review_encode(clean_review)
        encoded_review = keras.preprocessing.sequence.pad_sequences([encoded_review], value=self.word_index["<PAD>"], padding="post", maxlen=250)
        prediction = self.model.predict(encoded_review)
        string_prediction = str(prediction[0][0])

        return string_prediction


class ReviewCalification(Resource):
    def get(self, review):
        text_model = TextModel()
        prediction = text_model.predict_review(review)
        return prediction 
        
        """
        Predictions
            -The closer the prediction is to 1, the better the review
            -The closer the prediction is to 0, the worse the review
        """

api.add_resource(ReviewCalification,'/review/<review>')


if __name__ == '__main__':
    app.run(debug=True)