from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

class TextModel:
    def __init__(self):
        with open("model.pkl", "rb") as f:
            self.__model = pickle.load(f)
        with open("word_index.pkl", "rb") as f:
            self.word_index = pickle.load(f)
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
        return review.replace(',', "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").replace("/", "").strip().split(" ")

    def predict_review(self, review):
        clean_review = self.clean_review(review)
        encoded_review = self.review_encode(clean_review)
        encoded_review = np.pad(encoded_review, (0, 250 - len(encoded_review)), 'constant', constant_values=self.word_index["<PAD>"])
        prediction = self.model.predict(np.array([encoded_review]))
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
