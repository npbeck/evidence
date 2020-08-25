# Description
This repository belongs to the **research project 'Evidence'** by TU Darmstadt's Ubiquitious Knowledge Processing Lab.

The project aims to provide lexicographers with an interactive machine-learning-powered interface, easing and speeding up the process of choosing good example sentences, given a lemma.
- English project description: [Computer-assisted Interactive Extraction of Dictionary Examples from Large Corpora](https://www.informatik.tu-darmstadt.de/ukp/research_6/current_projects/ukp_evidence/ukp_project_evidence.en.jsp)
- German project description: [ EVIDENCE: Computer-unterstützte interaktive Extraktion guter Wörterbuchbeispiele aus großen Korpora](https://gepris.dfg.de/gepris/projekt/433249742)


# Content
This repository consists of two main components:
1. `flask_web_app/` - a web application designed to be used by lexicographers
2. `experiments/` - a couple of python libraries and scripts designed to be used for generating data representations (embeddings)
and then training machine-learning models with those representations

# Exemplary use case
Say you had a CSV file with annotated sentence examples in the following format:
`<example sentence>, <lemma>, <score>`. 
(We actually do not even require a `<lemma>` field, since we assume the lemma to be marked
in the example sentence using tags, e.g. `"This sentence's lemma is _&cow&_."`)

You could then load this CSV file and generate embeddings, i.e. fixed-size representations of the sentence and its lemma
using the `experiments/embeddings.py` library as well as the `experiments/embeddings_script.py` file.

Having generated the respective embeddings you could then move on to training a model with them. `experiments/log_reg.py` 
and its respective script provide you with a way of training a logistic regression, while `experiments/mlp.py` and its respective
script allow you to train a simple multilayer perceptron.