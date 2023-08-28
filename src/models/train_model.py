from gensim.models import KeyedVectors
import sent2vec
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer


def load_word2vec_model(model_path):
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model, None


def load_sent2vec_model(model_path):
    model = sent2vec.Sent2vecModel()
    model.load_model(model_path)
    return model, None


def load_bert_model():
    model_name = 'bert-base-uncased'
    model = BertModel.from_pretrained(model_name, return_dict=True)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_sbert_model():
    model_name = 'paraphrase-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    return model, None


def load_model(model_type, model_path=None):
    if model_type == 'word2vec':
        return load_word2vec_model(model_path)
    elif model_type == 'sent2vec':
        return load_sent2vec_model(model_path)
    elif model_type == 'bert':
        return load_bert_model()
    elif model_type == 'sbert':
        return load_sbert_model()
    else:
        raise Exception('Unsupported model')
