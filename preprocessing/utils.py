import glob
from tika import parser
import os
import re
import codecs

def read_html(txt_folder='../data/'):
  """
  Function reads html files in folder from argument
  Returns string where all txt content concatenated in one string
  Args:
      txt_folder (string): where data is located ex.: './path/'
  Returns:
      data (string): content from html folder
  """
  data = ''
  for html in glob.glob(os.path.join(txt_folder, '*.html')):
    with codecs.open(html, "rb", "windows-1251") as f:
      readed = f.read()
    readed = re.sub("[^А-Яа-я,.!?0-9]+", ' ', readed)
    data += readed
  return data

def read_txt(txt_folder='../data/'):
  """
  Function reads txt files in folder from argument
  Returns string where all txt content concatenated in one string
  Args:
      txt_folder (string): where data is located ex.: './path/'
  Returns:
      data (string): content from txt folder
  """
  data = ''
  for txt in glob.glob(os.path.join(txt_folder, '*.txt*')):
    with open(txt, 'r') as f:
      readed = f.read()
    readed = re.sub("[^А-Яа-я,.!?0-9]+", ' ', readed)
    data += readed
  return data

def read_pdf(pdf_folder='../data/'):
  """
  Function reads pdf files in folder from argument
  Returns string where all pdf content concatenated in one string
  Args:
      txt_folder (string): where data is located ex.: './path/'
  Returns:
      data (string): content from pdf folder
  """
  data = ''
  for pdf in glob.glob(os.path.join(pdf_folder, '*.pdf')):
    raw = parser.from_file(pdf)
    readed = raw['content']
    readed = re.sub("[^А-Яа-я,.!?0-9]+", ' ', readed)
    data += readed
  return data
