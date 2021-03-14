from nltk.corpus import stopwords
import copy
import pickle
from nltk.tokenize import sent_tokenize
import os
from navec import Navec
import numpy as np
import gensim
from gensim import utils
from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring

def my_save_word2vec_format(fname, vocab, vectors, binary=True, total_vec=2):
    """Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vectors : numpy.array
        The vectors to be stored.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
    total_vec : int, optional
        Explicitly specify total number of vectors
        (in case word vectors are appended with document vectors afterwards).

    """
    if not (vocab or vectors):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with utils.smart_open(fname, 'wb') as fout:
        print(total_vec, vector_size)
        fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary:
                row = row.astype(REAL)
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))
                
count = 0
russian_stopwords = stopwords.words('russian')
token_dict_init = {
                    'vecs': {
                              'right': {
                                          0: np.zeros((300,)),
                                          1: np.zeros((300,)),
                                          2: np.zeros((300,)),
                                          3: np.zeros((300,)),
                                          4: np.zeros((300,)),
                                          5: np.zeros((300,)),
                                          6: np.zeros((300,)),
                                          7: np.zeros((300,))
                                              },
                              'left': {
                                          0: np.zeros((300,)),
                                          1: np.zeros((300,)),
                                          2: np.zeros((300,)),
                                          3: np.zeros((300,)),
                                          4: np.zeros((300,)),
                                          5: np.zeros((300,)),
                                          6: np.zeros((300,)),
                                          7: np.zeros((300,))
                                              }
                              }
                    }

def w2v_convert(tokens_dict, book):
    w2v_convert_1 = {}
    w2v_convert_2 = {}
    w2v_convert_3 = {}
    w2v_convert_4 = {}
    w2v_convert_5 = {}
    w2v_convert_6 = {}
    w2v_convert_7 = {}
    w2v_convert_8 = {}
    for token in tokens_dict.keys():
        w2v_convert_8[token] = (tokens_dict[token]['vecs']['right'][0] + tokens_dict[token]['vecs']['left'][0] +\
                             tokens_dict[token]['vecs']['right'][1] + tokens_dict[token]['vecs']['left'][1] +\
                             tokens_dict[token]['vecs']['right'][2] + tokens_dict[token]['vecs']['left'][2] +\
                             tokens_dict[token]['vecs']['right'][3] + tokens_dict[token]['vecs']['left'][3] +\
                             tokens_dict[token]['vecs']['right'][4] + tokens_dict[token]['vecs']['left'][4] +\
                             tokens_dict[token]['vecs']['right'][5] + tokens_dict[token]['vecs']['left'][5] +\
                             tokens_dict[token]['vecs']['right'][6] + tokens_dict[token]['vecs']['left'][6] +\
                             tokens_dict[token]['vecs']['right'][7] + tokens_dict[token]['vecs']['left'][7]) / 16
        w2v_convert_7[token] = (tokens_dict[token]['vecs']['right'][0] + tokens_dict[token]['vecs']['left'][0] +\
                             tokens_dict[token]['vecs']['right'][1] + tokens_dict[token]['vecs']['left'][1] +\
                             tokens_dict[token]['vecs']['right'][2] + tokens_dict[token]['vecs']['left'][2] +\
                             tokens_dict[token]['vecs']['right'][3] + tokens_dict[token]['vecs']['left'][3] +\
                             tokens_dict[token]['vecs']['right'][4] + tokens_dict[token]['vecs']['left'][4] +\
                             tokens_dict[token]['vecs']['right'][5] + tokens_dict[token]['vecs']['left'][5] +\
                             tokens_dict[token]['vecs']['right'][6] + tokens_dict[token]['vecs']['left'][6]) / 14
        w2v_convert_6[token] = (tokens_dict[token]['vecs']['right'][0] + tokens_dict[token]['vecs']['left'][0] +\
                             tokens_dict[token]['vecs']['right'][1] + tokens_dict[token]['vecs']['left'][1] +\
                             tokens_dict[token]['vecs']['right'][2] + tokens_dict[token]['vecs']['left'][2] +\
                             tokens_dict[token]['vecs']['right'][3] + tokens_dict[token]['vecs']['left'][3] +\
                             tokens_dict[token]['vecs']['right'][4] + tokens_dict[token]['vecs']['left'][4] +\
                             tokens_dict[token]['vecs']['right'][5] + tokens_dict[token]['vecs']['left'][5]) / 12
        w2v_convert_5[token] = (tokens_dict[token]['vecs']['right'][0] + tokens_dict[token]['vecs']['left'][0] +\
                             tokens_dict[token]['vecs']['right'][1] + tokens_dict[token]['vecs']['left'][1] +\
                             tokens_dict[token]['vecs']['right'][2] + tokens_dict[token]['vecs']['left'][2] +\
                             tokens_dict[token]['vecs']['right'][3] + tokens_dict[token]['vecs']['left'][3] +\
                             tokens_dict[token]['vecs']['right'][4] + tokens_dict[token]['vecs']['left'][4]) / 10
        w2v_convert_4[token] = (tokens_dict[token]['vecs']['right'][0] + tokens_dict[token]['vecs']['left'][0] +\
                             tokens_dict[token]['vecs']['right'][1] + tokens_dict[token]['vecs']['left'][1] +\
                             tokens_dict[token]['vecs']['right'][2] + tokens_dict[token]['vecs']['left'][2] +\
                             tokens_dict[token]['vecs']['right'][3] + tokens_dict[token]['vecs']['left'][3]) / 8
        w2v_convert_3[token] = (tokens_dict[token]['vecs']['right'][0] + tokens_dict[token]['vecs']['left'][0] +\
                             tokens_dict[token]['vecs']['right'][1] + tokens_dict[token]['vecs']['left'][1] +\
                             tokens_dict[token]['vecs']['right'][2] + tokens_dict[token]['vecs']['left'][2]) / 6
        w2v_convert_2[token] = (tokens_dict[token]['vecs']['right'][0] + tokens_dict[token]['vecs']['left'][0] +\
                             tokens_dict[token]['vecs']['right'][1] + tokens_dict[token]['vecs']['left'][1]) / 4
        w2v_convert_1[token] = (tokens_dict[token]['vecs']['right'][0] + tokens_dict[token]['vecs']['left'][0]) / 2
    m = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=300)
    count = 1
    for version in [w2v_convert_1, w2v_convert_2, w2v_convert_3, w2v_convert_4, w2v_convert_5, w2v_convert_6, w2v_convert_7, w2v_convert_8]:
        w2v_convert = version
        m.vocab = w2v_convert
        m.vectors = np.array(list(w2v_convert.values()))
        my_save_word2vec_format(binary=True, fname=f'data/results/{book}_window_{count}.bin', total_vec=len(w2v_convert), vocab=m.vocab, vectors=m.vectors)
        count += 1

def get_w2v_book_representation(preprocessed_books, names, unks_remove=True):
    path = '../navec_hudlit_v1_12B_500K_300d_100q.tar'
    navec = Navec.load(path)
    count = 0
    for book in preprocessed_books:
        tokens_dict = {}
        for sentence in book:
            window = 8
            if unks_remove:
                words_padded = ['<pad>'] * window + [word for word in sentence if word \
                                                     in navec.vocab] + ['<pad>'] * window
            else:
                words_padded = ['<pad>'] * window + [word if word in navec.vocab \
                                                     else '<unk>' for word in sentence] + ['<pad>'] * window


            for word_idx in range(window, len(words_padded) - window):
                word = words_padded[word_idx]

                token_dict = copy.deepcopy(tokens_dict.get(word, None))
                if token_dict is None:
                    tokens_dict[word] = copy.deepcopy(token_dict_init)
                    token_dict = copy.deepcopy(tokens_dict.get(word, None))

                ###################### vectors part ######################
                for window_idx in range(window):
                    token_dict['vecs']['right'][window_idx] = (token_dict['vecs']['right'][window_idx] + navec[words_padded[word_idx + window_idx + 1]]) / 2
                    token_dict['vecs']['left'][window_idx] = (token_dict['vecs']['left'][window_idx] + navec[words_padded[word_idx - window_idx - 1]]) / 2
                ###################### vectors part ######################

                tokens_dict[word] = copy.deepcopy(token_dict)
        book_name = names[count]
        if not unks_remove:
            book_name += '_unks'
        w2v_convert(tokens_dict, book_name)
        count += 1
