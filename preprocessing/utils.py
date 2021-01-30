import re
import glob
import spacy
import ru2
from tika import parser
from gensim.models.phrases import Phrases, Phraser

def reading_txt(txt_folder):
    """
    Function reads txt files and replace double line breaks in it
    Returns string where all txt content concatenated in one string
    Args:
      txt_folder (string): where data is located ex.: './path/'
    Returns:
      data (string): content from txt folder
    """
    assert type(txt_folder) == str, 'invalid path in reading_txt function'
    assert len(txt_folder) > 0, 'invalid path in reading_txt function'
    assert txt_folder[-1] == '/', 'add "/" in argument of reading_txt function'

    data = ''
    for txt in glob.glob(txt_folder + '*.txt'):
        with open(txt, 'r') as f:
            readed = f.read()
        readed = readed.replace('\n\n', ' ')
        data += readed
    return data

def reading_pdf(pdf_folder):
    """
    Function reads pdf files and replace double line breaks in it
    Returns string where all pdf content concatenated in one string
    Args:
      pdf_folder (string): where data is located ex.: './path/'
    Returns:
      data (string): content from pdf folder
    """
    assert type(pdf_folder) == str, 'invalid path in reading_pdf function'
    assert len(pdf_folder) > 0, 'invalid path in reading_pdf function'
    assert pdf_folder[-1] == '/', 'add "/" in argument of reading_pdf function'

    data = ''
    for pdf in glob.glob(pdf_folder + '*.pdf'):
        raw = parser.from_file(pdf)
        readed = raw['content']
        readed = readed.replace('\n\n', ' ')
        data += readed
    return data

def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)

def cleaning_lemmatization(data, language='ru'):
    """
    Function reads string data, remove punctuation, lower and split text by dots
    Returns lemmatized and cleaned from punctuation list of strings
    Args:
        data (string): raw content from read functions
        language (string): what data we use for analysis
    Returns:
        cleaned (list): lemmatized and cleaned from punctuation list of strings
    """
    assert type(data) == str, 'invalid arg in cleaning_lemmatization function'
    assert type(language) == str, 'invalid arg in cleaning_lemmatization function'
    brief_cleaning = []
    nlp = ru2.load_ru2('./ru2')
    # nlp = spacy.load('ru2', disable=['tagger', 'parser', 'NER'])

    data = data.replace('ъ', '')
    data = data.replace('-', '')
    splitted_data = data.split('.')

    if language == 'ru':
      brief_cleaning = [re.sub("[^А-Яа-я]+", ' ', str(sentence)).lower() for sentence in splitted_data]
    elif language == 'en':
      brief_cleaning = [re.sub("[^A-Za-z]+", ' ', str(sentence)).lower() for sentence in splitted_data]
    txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]
    txt = [sentence for sentence in txt if sentence != None]
    txt = [sentence.replace('  ', '').split(' ') for sentence in txt]
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
