from sentence_transformers import SentenceTransformer

class Preprocessor:
    """ Class responsible for preprocessing the data. 
    Currently it only fetches mBert embeddings. """
    
    def __init__(self, model_name='distiluse-base-multilingual-cased'):
        """ Init with the appropriate name for the used (m)bert model """
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name) 

    def get_msbert_embedding(self, sentences):
        """ Fetches the embeddings of a list of given sentences """
        result = []
        for row in sentences:
            #TODO: match this to appropriate format; we probably don't have label.
            lemma, label, text, source = row
            ids = self.model.encode([text.replace('_&','').replace('&_','')])
            result.append([lemma,label,text,' '.join([str(i) for i in ids[0]])])
        return result

