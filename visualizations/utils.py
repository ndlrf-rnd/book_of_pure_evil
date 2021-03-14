import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import fasttext
from gensim.models import Word2Vec
import gensim

actions = ["смотреть", "видеть", "думать", "узнать", "догадаться", "чувствовать"]
objects = ["человек", "бог", "земля", "животное", "война", "мир"]
comparatives = ["сильный", "слабый", "хороший", "плохой", "мягкий", "твердый"]
concepts = ["добро", "зло", "лень", "сила", "трудно", "легко"]
ok = ["гнев", "боль"]

ru_keys = comparatives + ok + objects

def tsne_prep(model, keys=ru_keys):
  """
  Function gets gensim.w2v model and keys which should be used for viz
  Returns clusters with closest to keys neighbors
  Args:
      model (gensim.models.word2vec.Word2Vec): w2v model
      keys (list): list of strings's keyword which will be used for viz
  Returns:
      embedding_clusters (list): closest words's embeddings to initial keys
      word_clusters (list): closest words to initial keys
  """
  embedding_clusters = []
  word_clusters = []
  for word in keys:
      embeddings = []
      words = []
      if type(model) == fasttext.FastText._FastText:
        for _, similar_word in model.get_nearest_neighbors(word, k=30):
          words.append(similar_word)
          embeddings.append(model[similar_word])
          embedding_clusters.append(embeddings)
          word_clusters.append(words)
      elif type(model) == gensim.models.word2vec.Word2Vec:
        if word in model.vocab:
          for similar_word, _ in model.most_similar(word, topn=30):
              words.append(similar_word)
              embeddings.append(model[similar_word])
          embedding_clusters.append(embeddings)
          word_clusters.append(words)
      elif type(model) == gensim.models.keyedvectors.Word2VecKeyedVectors:
        if word in model.vocab:
          for similar_word, _ in model.most_similar(word, topn=30):
              words.append(similar_word)
              embeddings.append(model[similar_word])
          embedding_clusters.append(embeddings)
          word_clusters.append(words)
        
  return embedding_clusters, word_clusters

def tsne_plot_similar_words(labels, embedding_clusters, word_clusters, a=0.7, output_path='./', name_addition=''):
    """
    Function takes keys for w2v and embeddings takes from it
    Saves vizualization of embeddings in 2d
    Args:
        labels (list): list of keywords for vizualization
        embedding_clusters (list): list of embeddings for keywords's closest neighbors
        word_clusters (list): list of closest neighbors's words
        output_path (str): output path
    """
    plt.figure(figsize=(40, 30))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:,0]
        y = embeddings[:,1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.savefig(output_path + f"tsne_viz_{name_addition}.png", format='png', dpi=250, bbox_inches='tight')
