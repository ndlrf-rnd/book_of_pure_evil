from visualizations.utils import *
from preprocessing.utils import *
from split_utils import accum_txt_sentences
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
from context_vecs.make_w2v import get_w2v_book_representation
from sklearn.manifold import TSNE
import multiprocessing
import gc
import copy
import fasttext
import pandas as pd
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings(action='ignore')

def main():
    print("preparing data")
    pdf_goods = reading_pdf('data/')
    txt_goods = reading_txt('data/')
    pdf_goods_nums = reading_pdf('data/', keep_numbers=True)
    txt_goods_nums = reading_txt('data/', keep_numbers=True)
    data = copy.deepcopy(pdf_goods + txt_goods)
    #data = copy.deepcopy(pdf_goods_nums)
    del pdf_goods
    del txt_goods
    del pdf_goods_nums
    del txt_goods_nums
    gc.collect()
    
    print(len(data))
    tokenized = []
    for book in data:
        tokenized.append(sent_tokenize(book))
    cleaned = []
    for tokenized_book in tokenized:
        cleaned.append(cleaning_lemmatization(tokenized_book))
    #### for word2vec ####
    #sentences = w2v_bigram_prep(txt)
    #### for word2vec ####
    cores = multiprocessing.cpu_count()


    ################# book splitting sentences sum 3 #############
    #representation_way = accum_txt_sentences(3, txt)
    names = [path for path in os.listdir('data/') if '.pdf' in path or '.txt' in path]
    get_w2v_book_representation(cleaned, names)
    get_w2v_book_representation(cleaned, names, unks_remove=False)
    ################# book splitting sentences sum 3 #############


    ##################### getting visualization builded model #######################

    for book in names:
        w2v_model = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(f'data/results/{book}_unks_window_8.bin', binary=True)
        embedding_clusters, word_clusters = tsne_prep(w2v_model)
        tsne_model_en_2d = TSNE(perplexity=20, n_components=2, init='pca', n_iter=6000, random_state=32)
        embedding_clusters = np.array(embedding_clusters)
        n, m, k = embedding_clusters.shape
        embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
        tsne_plot_similar_words(ru_keys, embeddings_en_2d, word_clusters, name_addition=f'{book}_visualisation')


    ##################### getting visualization builded model #######################


    from graphviz import Digraph

    actions = ["смотреть", "видеть", "думать", "узнать", "догадаться", "чувствовать"]
    objects = ["человек", "бог", "земля", "животное", "война", "мир"]
    comparatives = ["сильный", "слабый", "хороший", "плохой", "мягкий", "твердый"]
    concepts = ["добро", "зло", "лень", "сила", "трудно", "легко"]

    set_names = ['actions', 'objects', 'comparatives', 'concepts']
    color_keys = ['coral1', 'cyan2', 'firebrick1', 'darkseagreen1', 'darksalmon', 'forestgreen', 'sienna1', 'pink', 'snow3']
    for book in names:
        count = 0
        for key_set in [actions, objects, comparatives, concepts]:
            w2v_model = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(f'data/results/{book}_unks_window_8.bin', binary=True)
            t = Digraph(f'ideas_with_{set_names[count]}_{book}', filename=f'data/results/ideas_with_{set_names[count]}_{book}.gv', engine='neato')

            t.attr('node', color='gold1', shape='circle', width='0.7')
            t.node(book)

            for keyword_idx in range(len(key_set)):
                if key_set[keyword_idx] in w2v_model.vocab:
                    t.attr('node', color=color_keys[keyword_idx], shape='circle')
                    t.node(key_set[keyword_idx])
                    t.edge(book, key_set[keyword_idx])

                    t.attr('node', shape='circle', fixedsize='true', width='1')
                    for i in w2v_model.most_similar([key_set[keyword_idx]]):
                        t.node(f'{i[0]}')
                        t.edge(key_set[keyword_idx], f'{i[0]}')
            count += 1

            t.attr(overlap='false')
            t.attr(label=r'Semantic proximity between core ideas\n'
                         f'in {book}')
            t.attr(fontsize='12')

            t.view()

#    print("preparing models\n")
#    w2v_model = Word2Vec(min_count=1,
#                        window=6,
#                        size=300,
#                        workers=cores-1)
#    w2v_model.build_vocab(sentences, progress_per=10000)
#    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
#    w2v_model.save('models/banned_books_w2v.model')
#    print("getting viz")
#    embedding_clusters, word_clusters = tsne_prep(w2v_model)
#    tsne_model_en_2d = TSNE(perplexity=20, n_components=2, init='pca', n_iter=6000, random_state=32)
#    embedding_clusters = np.array(embedding_clusters)
#    n, m, k = embedding_clusters.shape
#    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
#    tsne_plot_similar_words(ru_keys, embeddings_en_2d, word_clusters, name_addition='w2v_banned')
#
#    txt_fasttext = [' '.join(sentence) for sentence in txt]
#    df_train = pd.DataFrame({'txt': txt_fasttext})
#    df_train[['txt']].to_csv('train_data.txt', header=False, index=False, sep="\t")
#    model = fasttext.train_unsupervised('train_data.txt', model='skipgram')
#
#
#    model = fasttext.train_unsupervised('train_data.txt', model='cbow')
#    embedding_clusters, word_clusters = tsne_prep(model)
#    tsne_model_en_2d = TSNE(perplexity=40, n_components=2, init='pca', n_iter=6000, random_state=32)
#    embedding_clusters = np.array(embedding_clusters)
#    n, m, k = embedding_clusters.shape
#    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
#    tsne_plot_similar_words(ru_keys, embeddings_en_2d, word_clusters, name_addition='fasttext_cbow')
    

if __name__ == '__main__':
    main()
