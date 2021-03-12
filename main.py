from visualizations.utils import *
from preprocessing.utils import *
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
from sklearn.manifold import TSNE
import multiprocessing
import gc
import copy
import fasttext
import pandas as pd

def main():
    print("preparing data")
    pdf_goods = reading_pdf('data/')
    txt_goods = reading_txt('data/')
    data = pdf_goods + txt_goods
    data = copy.deepcopy(pdf_goods)
    del pdf_goods
    del txt_goods
    gc.collect()
    
    print(len(data))
    sentences = []
    for book in data:
        sentences.extend(sent_tokenize(book))
    txt = cleaning_lemmatization(sentences)
    sentences = w2v_bigram_prep(txt)
    cores = multiprocessing.cpu_count()


    print("preparing models\n")
    w2v_model = Word2Vec(min_count=1,
                        window=6,
                        size=300,
                        workers=cores-1)
    w2v_model.build_vocab(sentences, progress_per=10000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    w2v_model.save('models/banned_books_w2v.model')
    print("getting viz")
    embedding_clusters, word_clusters = tsne_prep(w2v_model)
    tsne_model_en_2d = TSNE(perplexity=20, n_components=2, init='pca', n_iter=6000, random_state=32)
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    tsne_plot_similar_words(ru_keys, embeddings_en_2d, word_clusters, name_addition='w2v_banned')

    txt_fasttext = [' '.join(sentence) for sentence in txt]
    df_train = pd.DataFrame({'txt': txt_fasttext})
    df_train[['txt']].to_csv('train_data.txt', header=False, index=False, sep="\t")
    model = fasttext.train_unsupervised('train_data.txt', model='skipgram')

    embedding_clusters, word_clusters = tsne_prep(model)
    tsne_model_en_2d = TSNE(perplexity=40, n_components=2, init='pca', n_iter=6000, random_state=32)
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    tsne_plot_similar_words(ru_keys, embeddings_en_2d, word_clusters, name_addition='fasttext_skipgram')

    model = fasttext.train_unsupervised('train_data.txt', model='cbow')
    embedding_clusters, word_clusters = tsne_prep(model)
    tsne_model_en_2d = TSNE(perplexity=40, n_components=2, init='pca', n_iter=6000, random_state=32)
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    tsne_plot_similar_words(ru_keys, embeddings_en_2d, word_clusters, name_addition='fasttext_cbow')
    

if __name__ == '__main__':
    import nltk
    nltk.download('punkt')
    main()
