"""
author: Nils Beck, nils.beck@pm.me

Evidence Project
@ Ubiquitous Knowledge Processing Lab
@ TU Darmstadt, Germany

July 2020

In this file I generate the embeddings that are to be used in training models afterwards
"""
import experiments.embeddings as emb

# BEGIN SCRIPT #
SOURCE_PATH = '/ukp-storage-1/nbeck/evidence/data/gbe-examples.csv'
TARGET_PATH = '/ukp-storage-1/nbeck/evidence/data/'

_, sentences, _ = emb.extract_data_from_csv(SOURCE_PATH)
emb.generate_embeddings(sentences)