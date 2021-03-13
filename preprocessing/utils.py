import re
import glob
import spacy
import fitz
import os
from tika import parser
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, Phraser
from deeppavlov import build_model, configs

def reading_txt(txt_folder='../data/'):
    """
    Function reads txt files and replace double line breaks in it
    Returns string where all txt content concatenated in one string
    Args:
      txt_folder (string): where data is located ex.: './path/'
    Returns:
      data (string): content from txt folder
    """
    books = []
    for txt in glob.glob(os.path.join(txt_folder, '*.txt')):
        print(txt)
        data = ''
        try:
            with open(txt, 'r') as f:
                readed = f.read()
                readed = re.sub("[^А-Яа-я\ \t,.!?]+", ' ', readed)
                data += ' ' + readed
            books.append(data)
        except:
            print(f'error occured while reading {txt}')
    return books

def reading_pdf(pdf_folder='../data/'):
    """
    Function reads pdf files and replace double line breaks in it
    Returns string where all pdf content concatenated in one string
    Args:
      txt_folder (string): where data is located ex.: './path/'
    Returns:
      data (list): content from txt folder split by books
    """
    assert type(pdf_folder) == str, 'invalid path in reading_pdf function'
    assert len(pdf_folder) > 0, 'invalid path in reading_pdf function'
    books = []
    for pdf in glob.glob(os.path.join(pdf_folder, '*.pdf')):
        print(pdf)
        data = ''
        pdf_document = pdf 
        doc = fitz.open(pdf_document)
        print ("number of pages: %i" % doc.pageCount)
        for page_num in range(doc.pageCount):
            page1 = doc.loadPage(page_num)
            page1text = page1.getText("text")
            page1text = re.sub("[^А-Яа-я\ \t,.!?]+", ' ', page1text)
            data += ' ' + page1text + ' '
        books.append(data)
    return books

def cleaning_lemmatization(splitted_book):
    """
    Function reads splitted by chunks book, remove punctuation, lower 
    Returns lemmatized and cleaned from punctuation list of list of tokens
    """
    txt = []
    russian_stopwords = stopwords.words('russian')
    print("building morphological parser model")

    morph_model = build_model(configs.morpho_tagger.BERT.morpho_ru_syntagrus_bert, download=True)
#    for chunk_id in range(len(splitted_book)):
#        chunk_len = len(splitted_book[chunk_id])
#        if chunk_len > 512:
#            parts_num = chunk_len // 512
#            splitted_chunk = splitted_book.pop(chunk_id).split()
#            part_idx = len(splitted_chunk) // parts_num
#            while parts_num > 0:
#                splitted_book.extend([' '.join(splitted_chunk[(parts_num - 1) * part_idx : parts_num * part_idx])])
#                parts_num -= 1
    for morph_line in morph_model.batched_call(splitted_book, batch_size=32):
        sentence = []
        for line in morph_line.split('\n'):
            if len(line) > 3:
                parsed_token = line.split()
                word = parsed_token[2]
                token_type = parsed_token[3]
                if len(word) > 2 and word not in russian_stopwords and token_type != "PUNCT":
                    sentence.append(word)
        txt.append(sentence)
    return txt


def w2v_bigram_prep(data):
    """
    Function reads list of cleaned data, make bigrams with gensim funcs
    Returns generator in class: 'gensim.interfaces.TransformedCorpus'
    Args:
        data (list): cleaned list of strings from text
    Returns:
        transformed_corpus (gensim.interfaces.TransformedCorpus): generator compatible with gensim.word2vec
    """
    assert type(data) == list, 'invalid arg in w2v_bigram_prep function'

    phrases = Phrases(data, min_count=1, progress_per=10000)
    bigram = Phraser(phrases)
    transformed_corpus = bigram[data]
    return transformed_corpus
