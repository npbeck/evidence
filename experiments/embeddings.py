"""
author: Nils Beck, nils.beck@pm.me

Evidence Project
@ Ubiquitous Knowledge Processing Lab
@ TU Darmstadt, Germany

July 2020

This file contains all functionality regarding the generation of embeddings,
which are then used to train machine learning models
"""
import csv
from time import strftime, gmtime

import numpy                                            # version 1.18.2
import torch                                            # version 1.5.0
import logging
from sentence_transformers import SentenceTransformer   # version 0.2.5.1
from transformers import XLMRobertaTokenizer, \
    XLMRobertaModel                                     # version 2.3.0


# this class allows for a simple way of documenting and displaying the amount of sentences longer than 512 tokens and
# thus not completely represented in the embeddings
class LongSentenceCounter:
    total = 0

    def inc(self):
        """
        increment the counter variable by one
        """
        self.total += 1

    def show(self):
        """
        log the counter's accumulated information
        """
        if self.total > 0:
            logging.info(str(self.total) + ' of ' + str(len(sentences)) + ' sentences account for more than 512 tokens '
                                                                          'respectively. Since our model can take at '
                                                                          'most 512 tokens as input, only a portion of '
                                                                          'each of these sentences was considered for '
                                                                          'calculating the embeddings.')

    def reset(self):
        """
        reset the counter variable to zero
        """
        self.total = 0


def _extract_lemmas_from_sentences(sentences):
    """
    extracts the first occurrence of a lemma from each sentence, using the markers
    :param sentences: list or array of strings representing the DIRTY sentences, i.e. with markers for the lemmas
    :return: a list of the lemmas that were found, corresponding in position to the given sentence (i.e. one lemma per
    sentence)
    """
    lemmas = []
    for s in sentences:
        # extract first occurrence of lemma
        i = s.index('_&')
        j = s.index('&_')
        lemmas.append(s[i + 2:j])
    return lemmas


def _shorten_ids(sentence_ids, lemma_index, lemma_length, lsc):
    """
    ensure that the given ids do NOT surpass a length of 512, since our model will discard anything tokens beyond that
    length. If such a length is surpassed, ensure that the lemma remains amongst the 512 tokens we cut out
    :param lsc: reports amount of sentences that are too long for being entirely tokenized
    :param sentence_ids: ids of all sentence tokens
    :param lemma_index: index at which lemma id(s) start/is
    :param lemma_length: amount of lemma ids
    :return: list of token ids, containing lemma ids and being at most 512 ids long
    """
    if len(sentence_ids) <= 512:
        return sentence_ids

    # shorten the sentence_ids array, placing given lemma_ids as much into the center of the shortened array as possible
    # i.e. [<part_a> <lemma_ids> <part_b>] with parts a and b as much as possible of same length
    else:
        logging.info('A sentence was found to be longer than 512 tokens and is thus being shortened to a length of 512.')
        # increment the LongSentenceCounter instance
        lsc.inc()
        ideal_rest_length = int(numpy.floor((512 - lemma_length) / 2))
        left_rest_len = lemma_index
        right_rest_len = len(sentence_ids) - (lemma_index + lemma_length)

        if left_rest_len > ideal_rest_length and right_rest_len > ideal_rest_length:
            # shorten both to ideal length
            return sentence_ids[lemma_index - ideal_rest_length: lemma_index + lemma_length + ideal_rest_length]
        elif left_rest_len > ideal_rest_length:
            # shorten only left side
            return sentence_ids[lemma_index - (512 - lemma_length - right_rest_len):]
        elif right_rest_len > ideal_rest_length:
            # shorten only right side
            return sentence_ids[:lemma_index + lemma_length + 512 - lemma_length - left_rest_len]


def _generate_lemma_embs_with_special_tokens(sentences, model, tokenizer, lsc):
    """
    generate contextualized lemma word embeddings, using the SPECIAL TOKENS method to ensure uniform length,
    i.e. all lemmas are added to the 'additional_special_tokens' list, preventing multi-token lemmas from occurring
    and hence ensuring lemma embeddings to be of uniform length.
    :param model: model to be used to calculate the embeddings
    :param tokenizer: tokenizer to be used for tokenizing the sentences
    :param lsc: reports amount of sentences that are too long for being entirely tokenized
    :param sentences: list or array of strings representing the DIRTY sentences, i.e. with markers for the lemmas
    :return: contextualized lemma word embeddings
    """
    inline_lemmas = _extract_lemmas_from_sentences(sentences)

    # clean sentences
    clean_sentences = []
    for dirty_sentence in sentences:
        clean_sentences.append(dirty_sentence.replace('_&', ' ').replace('&_', ' '))

    lemma_embeddings = []
    for sentence, lemma in zip(clean_sentences, inline_lemmas):
        lemma_id = tokenizer.encode(lemma)[0]
        sentence_ids = tokenizer.encode(sentence)
        shortened_sentence_ids = _shorten_ids(sentence_ids, sentence_ids.index(lemma_id),
                                              1, lsc)  # map sentence to list of IDs representing it

        # generate contextualized embeddings,
        # unsqueeze IDs to get batch size of 1 as added dimension
        with torch.no_grad():
            embeddings = model(input_ids=torch.tensor(shortened_sentence_ids).unsqueeze(0))[0]

        # cut the lemma embeddings out of these sentence embeddings, convert them into numpy array
        current_lemma_embeddings = embeddings[0][sentence_ids.index(lemma_id)].detach().numpy()

        # add current lemma embeddings to the list
        lemma_embeddings.append(current_lemma_embeddings)

    return lemma_embeddings


def _generate_lemma_embs_with_mean(sentences, model, tokenizer, lsc):
    """
    generate contextualized lemma word embeddings, using the MEAN method to ensure uniform length,
    i.e. multi-token lemma embeddings are calculated token by token and then reduced to one embedding
    by calculating their arithmetic mean
    :param model: model to be used to calculate the embeddings
    :param tokenizer: tokenizer to be used for tokenizing the sentences
    :param lsc: reports amount of sentences that are too long for being entirely tokenized
    :param sentences: list or array of strings representing the DIRTY sentences, i.e. with markers for the lemmas
    :return: contextualized lemma word embeddings
    """
    lemma_embeddings = []
    for dirty_sentence in sentences:
        # extract first occurrence of lemma
        i = dirty_sentence.index('_&')
        j = dirty_sentence.index('&_')
        assert (i < j)
        lemma = dirty_sentence[i + 2:j]

        # clean up the sentence
        clean_sentence = dirty_sentence.replace('_&', ' ').replace('&_', ' ')

        # map sentence and lemma to list of IDs representing it
        lemma_ids = tokenizer.encode(lemma)[1:-1]
        sentence_ids = tokenizer.encode(clean_sentence)

        # find the index of the lemma ids in the sentence ids
        lemma_index = -1
        for i in range(len(sentence_ids)):
            for j in range((len(lemma_ids))):
                if sentence_ids[i + j] != lemma_ids[j]:
                    break
                if j == len(lemma_ids) - 1:
                    lemma_index = i
        assert lemma_index > -1, 'Lemma: ' + lemma + '. Sentence: ' + clean_sentence + \
                                 ' \n Lemma IDS: ' + str(lemma_ids) + ' Sentence IDS: ' + str(sentence_ids)

        sentence_ids = _shorten_ids(sentence_ids, lemma_index, len(lemma_ids), lsc)

        # generate contextualized embeddings,
        # un-squeeze IDs to get batch size of 1 as added dimension
        with torch.no_grad():
            embeddings = model(input_ids=torch.tensor(sentence_ids).unsqueeze(0))[0]

        # cut the lemma embeddings out of these sentence embeddings, convert them into numpy array
        current_lemma_embeddings = embeddings[0][lemma_index: lemma_index + len(lemma_ids)].detach()

        try:
            # calculate the arithmetic mean on the resulting embedding vectors
            if len(lemma_ids) > 1:
                base_tensor = current_lemma_embeddings[0]
                for i in range(len(current_lemma_embeddings))[1:]:
                    base_tensor.add(current_lemma_embeddings[i])
                current_lemma_embeddings = torch.div(base_tensor,
                                                     len(current_lemma_embeddings))  # possibly better to use true_divide
        except IndexError:
            logging.error('Index error. Lemma IDs: ' + str(lemma_ids) + '. Sentence: ' + dirty_sentence)

        # add current lemma embeddings to the list
        lemma_embeddings.append(current_lemma_embeddings.numpy())

    return lemma_embeddings


def _has_valid_syntax(row):
    """
    Check whether a given row in the CSV file has the required syntax, i.e.
    - lemma is an existing string object
    - example sentence is a string and contains at least one pair of lemma markers in the right order
    - score is one of the following strings: '0', '1', '2', '3', '4'
    :param row: a dictionary containing all values in a CSV row
    """
    lemma = row['Lemma']
    sentence = row['Example']
    score = row['Score']

    # check lemma
    if lemma is None or type(lemma) != str:
        return False

    # check sentence
    if sentence is None or type(sentence) != str or not sentence.__contains__('_&') or not sentence.__contains__('&_'):
        return False
    i = sentence.index('_&')
    j = sentence.index('&_')
    if i >= j:
        return False

    # check score
    valid_scores = ['0', '1', '2', '3', '4']
    if score is None or type(score) != str or score not in valid_scores:
        return False

    return True


def extract_data_from_csv(path):
    """
    extracts example sentences, the lemmas whose use they exemplify, and their labels
    from a given csv file
    only valid sentences are returned, i.e. those satisfying certain conditions
    :param path: path to the desired file
    :return: <lemmas>, <"dirty" sentences>, <labels>, i.e. three arrays
    """
    lemmas = []  # a string array containing the lemmas
    sentences = []  # a string array containing the sentences
    labels = []  # a int array containing their corresponding scores

    faulty_samples_counter = 0

    with open(path, 'r', encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # only add a row if it has the syntax we expect it to have
            if _has_valid_syntax(row):
                lemmas.append(str(row['Lemma']))
                sentences.append(str(row['Example']))
                labels.append(int(row['Score']))
            else:
                faulty_samples_counter += 1

        if faulty_samples_counter > 0:
            logging.info(str(faulty_samples_counter) + ' faulty samples were found among the given ' +
                         str(len(sentences)) + '. They will not be considered in the embeddings calculations.')

    # sort by sentence length (ascending)
    # we zip and unzip using the same function
    return zip(*sorted(zip(lemmas, sentences, labels), key=lambda x: len(x[1])))


def generate_embeddings(sentences, method='special_tokens', target_dir_path='/ukp-storage-1/nbeck/evidence/data/',
                        batch_size=500, logging_level=logging.INFO):
    """
    Base method for generating embeddings.
    Handles the batch-wise calculation of them, thus enabling more efficient calculations in case errors bring a given
    batch to halt.
    Resulting embeddings are stored in a file in the given directory.
    :param logging_level: desired level of logging (from logging library)
    :param batch_size: amount of sentences to be processes in one run/batch
    :param sentences: list or array of strings representing the DIRTY sentences, i.e. with markers for the lemmas
    :param method: how to calculate fixed-size word embeddings for the lemma (which might be tokenized as multiple tokens
    can be 'special_tokens' or 'mean'
    :param target_dir_path: path of the directory where embeddings are supposed to be stored
    """
    # set up the logger
    logging.basicConfig(level=logging_level,
                        format='%(asctime)s - %(message)s')

    logging.info('Beginning the calculation of embeddings.')
    # create an empty file
    file_name = 'word_plus_sen_embeddings_' + method + '_' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.npy'
    numpy.save(target_dir_path + file_name, [])

    # load sentence embeddings model (just) once rather than in every batch loop
    sen_embd_model = SentenceTransformer('distiluse-base-multilingual-cased')

    # load lemma embeddings model (just) once rather than every time the respective sub-function is called
    lemma_embd_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
    lemma_embd_model.eval()

    # load lemma embeddings tokenizer (just) once - you get it
    lemma_embd_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    if method == 'special_tokens':
        lemma_embd_tokenizer = XLMRobertaTokenizer.\
            from_pretrained('xlm-roberta-base', additional_special_tokens=_extract_lemmas_from_sentences(sentences))

    # create a LongSentenceCounter
    lsc = LongSentenceCounter()

    # iterate over batches
    for i in range(int(numpy.ceil(len(sentences) / batch_size))):
        # determine batch length
        batch_len = numpy.min([batch_size, len(sentences) - i * batch_size])
        # determine current batch
        current_batch = sentences[i * batch_size: i * batch_size + batch_len]
        # generate contextualized word embedding for each lemma
        if method == 'special_tokens':
            lemma_embeddings = _generate_lemma_embs_with_special_tokens(current_batch, lemma_embd_model, 
                                                                        lemma_embd_tokenizer, lsc)
        elif method == 'mean':
            lemma_embeddings = _generate_lemma_embs_with_mean(current_batch, lemma_embd_model, 
                                                              lemma_embd_tokenizer, lsc)
        else:
            raise Exception('Specified method of dealing with multi-token lemma not found')

        # generate sentence embeddings for sentences
        sen_embeddings = sen_embd_model.encode(current_batch)

        # concatenate sentence and word embeddings
        current_embds = []
        for a, b in zip(sen_embeddings, lemma_embeddings):
            current_embds.append(numpy.append(a, b))

        # store/append embeddings to file
        arr = numpy.load(target_dir_path + file_name, allow_pickle=True)
        new_embds = numpy.append(arr, current_embds)
        numpy.save(target_dir_path + file_name, new_embds)
        # log that current batch was stored successfully
        logging.info('Batch #' + str(i) + ' of embeddings successfully stored in ' + target_dir_path + file_name + '.')

    logging.info('ALL EMBEDDINGS CALCULATED SUCCESSFULLY!')
    lsc.show()
    lsc.reset()

    # reshape embeddings, so they can be more intuitively used (up to this point they are one flat list)
    arr = numpy.load(target_dir_path + file_name, allow_pickle=True)
    arr = arr.reshape(int(len(arr) / 1280), 1280)
    numpy.save(target_dir_path + file_name, arr)
