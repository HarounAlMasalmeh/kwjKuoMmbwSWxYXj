import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def average_word2vec(model, sentence):
    words = sentence.split()
    word_vectors = [model[word] for word in words if word in model.index_to_key]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


def get_word2vec_embeddings(model, sentences):
    vectors = [average_word2vec(model, sentence) for sentence in sentences]
    return vectors


def get_sent2vec_embeddings(model, sentences):
    vectors = model.embed_sentences(sentences)
    return vectors


def get_bert_embeddings(model, sentences, tokenizer):
    """Return embeddings for a list of sentences"""
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.pooler_output
    return embeddings


def get_sbert_embeddings(model, sentences):
    embeddings = model.encode(sentences)
    return embeddings


def get_embeddings(model_name, model, sentences, tokenizer=None):
    if model_name == 'word2vec':
        return get_word2vec_embeddings(model, sentences)
    elif model_name == 'sent2vec':
        return get_sent2vec_embeddings(model, sentences)
    elif model_name == 'bert':
        if tokenizer is None:
            raise Exception('You need to provide the tokenizer for bert model')
        return get_bert_embeddings(model, sentences, tokenizer)
    elif model_name == 'sbert':
        return get_sbert_embeddings(model, sentences)
    else:
        raise Exception('Unsupported model')


def get_similarities(sentence_embeddings1, sentence_embeddings2):
    similarities = cosine_similarity(sentence_embeddings1, sentence_embeddings2)
    return similarities


