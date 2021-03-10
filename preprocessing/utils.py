import re
import glob
import spacy
import ru2
import fitz
import os
from tika import parser
from gensim.models.phrases import Phrases, Phraser

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
                readed = re.sub("[^А-Яа-я\ \t\n,.!?]+", '', readed)
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
            page1text = re.sub("[^А-Яа-я\ \t\n,.!?]+", '', page1text)
            data += ' ' + page1text + ' '
        books.append(data)
    return books

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
