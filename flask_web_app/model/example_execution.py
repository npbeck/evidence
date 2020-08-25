from preprocessor import Preprocessor
from logistic_regression_classifier import Logistic_Regression_Classifier
import csv

def read_data(infile):
    sentences = []
    with open(infile,'r') as lines:
        reader = csv.reader(lines, delimiter=',', quotechar='"') 
        for row in reader:
            lemma, label, text, source = row
            sentences.append((lemma, label, text, source))
    return sentences
    
    
preprocessing = Preprocessor()
classifier = Logistic_Regression_Classifier()

sentences = read_data('example_data/test.csv')
embeddings = preprocessing.get_msbert_embedding(sentences)

predictions = classifier.get_prediction_with_sentences(embeddings)

print(predictions)
