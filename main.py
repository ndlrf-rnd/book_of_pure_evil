from visualizations.utils import *
from preprocessing.utils import *
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import multiprocessing
import gc

def main():
    print("preparing data")
    pdf_goods = reading_pdf('data/')
    txt_goods = reading_txt('data/')
    data = pdf_goods + txt_goods
    del pdf_goods
    del txt_goods
    gc.collect()
    txt = cleaning_lemmatization(data)
    del data
    gc.collect()
    sentences = w2v_bigram_prep(txt)
    del txt
    gc.collect()
    cores = multiprocessing.cpu_count()
    
    print("preparing models\n if you have nothing in data folder, then you got mistake in w2v.train method")
    w2v_model = Word2Vec(min_count=5,
                        window=6,
                        size=300,
                        workers=cores-1)
    w2v_model.build_vocab(sentences, progress_per=10000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    w2v_model.save('models/banned_books_w2v.model')
    print("getting viz")
    embedding_clusters, word_clusters = tsne_prep(w2v_model)
    tsne_model_en_2d = TSNE(perplexity=40, n_components=2, init='pca', n_iter=6000, random_state=32)
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    tsne_plot_similar_words(ru_keys, embeddings_en_2d, word_clusters)

if __name__ == '__main__':
    main()
