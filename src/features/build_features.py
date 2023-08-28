import pandas as pd


def build_features(file_path):
    df = pd.read_csv(file_path, index_col='id')
    title_df = df['job_title']
    sentences = list(set(title_df.values))

    return sentences
