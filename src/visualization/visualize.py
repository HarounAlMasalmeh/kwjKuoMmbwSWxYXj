from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        x, y = reduced_embeddings[i]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x+0.01, y, label, fontsize=9)
    plt.show()
