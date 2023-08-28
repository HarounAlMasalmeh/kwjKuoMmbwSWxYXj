import sys
import argparse
from src.features.build_features import build_features
from src.models.predict_model import get_similarities, get_embeddings
from src.models.train_model import load_model


def get_top_10(keywords, sentences, similarities):
    top_10_similar_sentences = []

    for i, sentence in enumerate(keywords):
        # Getting the indices of top 10 similar sentences
        top_10_indices = similarities[i].argsort()[-10:][::-1]

        top_10_for_sentence = [(sentences[j], similarities[i][j]) for j in top_10_indices]
        top_10_similar_sentences.append((sentence, top_10_for_sentence))

    # Printing the results
    for s1, top_10 in top_10_similar_sentences:
        print(f"Most similar sentences to '{s1}':")
        for s2, sim in top_10:
            print(f"   '{s2}' with a similarity of {sim:.4f}.")
        print("---------")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A simple command line arguments parser")

    parser.add_argument("-d", "--data", type=str, default="../data/raw/potential-talents.csv", help="data file path")
    parser.add_argument("-n", "--name", type=str, default="sbert", help="model name")
    parser.add_argument("-m", "--model", type=str, default=None, help="model file path")

    args = parser.parse_args()

    print(args)

    data_file_path = args.data
    model_name = args.name
    model_path = args.model

    sentences = build_features(data_file_path)
    keywords = ["Aspiring human resources", "seeking human resources"]

    model, tokenizer = load_model(model_name, model_path)
    keywords_embeddings = get_embeddings(model_name, model, keywords, tokenizer)
    sentences_embeddings = get_embeddings(model_name, model, sentences, tokenizer)

    similarities = get_similarities(keywords_embeddings, sentences_embeddings)
    print(f"Model: {model_name}")
    get_top_10(keywords, sentences, similarities)




