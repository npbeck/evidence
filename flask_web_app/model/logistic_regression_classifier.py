""" Logistic Regression Classifier for classifying good example sentences."""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

class Logistic_Regression_Classifier:
    def __init__(self, model="pretrained_models/lr_model.pkl" ):
        self.model_path = model
        self.model = pickle.load(open(self.model_path, 'rb'))

    def get_features_only(self, data):
        """Get X-data."""
        dx = []
        for row in data:
            lemma,label,text,feature = row
            x = dx.append(np.fromstring(feature, sep=' '))
        return dx
        
    def get_features(self, data):
        """Get X-data and some."""
        lemmas, labels, texts, dx = [],[],[],[]
        for row in data:
            lemma,label,text,feature = row
            lemmas.append(lemma)
            labels.append(label)
            texts.append(text)
            x = dx.append(np.fromstring(feature, sep=' '))
        return lemmas, labels, texts, dx

    def get_prediction(self, data):
        """Get prediction from preprocessed data. """
        X_test = self.get_features(data)
        pred = self.model.predict(X_test)

        return pred

    def get_prediction_with_sentences(self, data):
        """Get prediction from preprocessed data. """
        lemmas, labels, texts, X_test = self.get_features(data)
        pred = self.model.predict(X_test)

        return [(lemmas[i],labels[i],texts[i],pred[i]) for i in range(len(lemmas))]

