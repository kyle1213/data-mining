import os
import bs4
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from readability.readability import Unparseable
from readability.readability import Document
import nltk
from nltk import sent_tokenize
from nltk import wordpunct_tokenize
from nltk import pos_tag
import time
import string
from nltk.text import TextCollection
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import collections


#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

CAT_PATTERN = r'([a-z_\s]+)/.*'
DOC_PATTERN = r'(?!\.)[a-z_\s]+/[0-9]+\.html'
TAGS = []
title_TAGS = ['h1']
abstract_TAGS = ['blockquote']

#html_list_path = os.getcwd() + '/html_path_list_post_process.txt'
#html_path = os.getcwd() + '/arxiv'
#with open(html_list_path, 'r') as f:
#    lines = f.readlines()
#for i, line in enumerate(lines):
#    lines[i] = line.strip()
#print("length of lines(htmls from scraping): ", len(lines))


#####1
def html_to_text(path, TAGS):
    path = 'arxiv/' + path
    print(path)
    with open(path, 'r', encoding='UTF-8') as f:
        html = f.read()
    soup = bs4.BeautifulSoup(html, "lxml")
    for tag in soup.find_all(TAGS):
        yield tag.get_text()

"""
for i, path in enumerate(lines):
    print(i, path)
    x = html_to_text(path)
    if i >5:
        break
print(list(x))
"""
#####1


#####2
class HTMLCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
    A corpus reader for raw HTML documents to enable preprocessing.
    """

    def __init__(self, root, fileids=DOC_PATTERN, encoding='utf8',
                 tags=TAGS, **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining
        arguments are passed to the ``CorpusReader`` constructor.
        """
        # Add the default category pattern if not passed into the class.
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        # Initialize the NLTK corpus reader objects
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids, encoding)

        # Save the tags that we specifically want to extract.
        self.tags = tags

    def resolve(self, fileids, categories):
        """
        Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function. Implemented similarly to
        the NLTK ``CategorizedPlaintextCorpusReader``.
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Returns the complete text of an HTML document, closing the document
        after we are done reading it and yielding it in a memory safe fashion.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path in self.abspaths(fileids):
            with open(path, 'r', encoding='UTF-8') as f:
                yield f.read()

    def sizes(self, fileids=None, categories=None):
        """
        Returns a list of tuples, the fileid and size on disk of the file.
        This function is used to detect oddly large files in the corpus.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, getting every path and computing filesize
        for path in self.abspaths(fileids):
            yield os.path.getsize(path)


title_corpus = HTMLCorpusReader('', CAT_PATTERN, DOC_PATTERN, tags=title_TAGS)
title_fileids = title_corpus.fileids()
title_documents = title_corpus.docs(categories=title_corpus.categories())
title_htmls = list(title_documents)

abstract_corpus = HTMLCorpusReader('', CAT_PATTERN, DOC_PATTERN, abstract_TAGS)
abstract_fileids = abstract_corpus.fileids()
abstract_documents = abstract_corpus.docs(categories=abstract_corpus.categories())
abstract_htmls = list(abstract_documents)
#####2


def html(documents):
    for doc in documents:
        try:
            yield Document(doc).summary()
        except Unparseable as e:
            print("Could not parse HTML: {}".format(e))
            continue


#####3
def paras(htmls, TAGS): #paragraph로 나누기
    for html in htmls:
        soup = bs4.BeautifulSoup(html, 'lxml')
        for element in soup.find_all(TAGS):
            yield element.text
        soup.decompose()


title_paragraphs = list(paras(title_htmls, title_TAGS))
temp_title_paragraphs = []
for para in title_paragraphs:
    if "Title:" in para: # and len(para)>30
        temp_title_paragraphs.append(para.strip('Title:\n'))
title_paragraphs = temp_title_paragraphs
print("title_paragraphs len: ", len(title_paragraphs))

abstract_paragraphs = list(paras(abstract_htmls, abstract_TAGS))
print("abstract_paragraphs len: ", len(abstract_paragraphs), "\n")

#print(title_paragraphs[:3])
#print(abstract_paragraphs[:3])
#####3


def sents(paragraph):
    for sentence in sent_tokenize(paragraph):
        yield sentence


def words(paragraph):
    for sentence in sents(paragraph):
        for word in wordpunct_tokenize(sentence):
            yield word


#####4
title_categories = title_corpus.categories()
abstract_categories = abstract_corpus.categories()


def describe(paragraphs, fileids, categories):
    started = time.time()

    counts = nltk.FreqDist()
    tokens = nltk.FreqDist()

    for para in paragraphs:
        counts['paras'] += 1

        for sent in nltk.sent_tokenize(para):
            counts['sents'] += 1

            for word in nltk.wordpunct_tokenize(sent):
                counts['words'] += 1
                tokens[word] += 1

    n_fileids = len(fileids)
    n_topics = len(categories)

    return {
        'files':  n_fileids,
        'topics': n_topics,
        'paragraphs':  counts['paras'],
        'sentences':  counts['sents'],
        'words':  counts['words'],
        'vocabulary size':  len(tokens),
        'lexical diversity': float(counts['words']) / float(len(tokens)),
        'paragraphs per document':  float(counts['paras']) / float(n_fileids),
        'sentences per paragraph':  float(counts['sents']) / float(counts['paras']),
        'secs':   time.time() - started,
    }


print("descreibe title_paragraphs", describe(title_paragraphs, title_fileids, title_categories))
print("descreibe abstract_paragraphs", describe(abstract_paragraphs, abstract_fileids, abstract_categories), "\n")

papers_list = []
for key, value in zip(title_paragraphs, abstract_paragraphs):
    temp_dict = dict()
    temp_dict['title'] = key
    temp_dict['abstract'] = value
    papers_list.append(temp_dict)
#####4


def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation:
            continue
        yield stem.stem(token)

"""
def nltk_tfidf_vectorize(corpus):

    tokenized_corpus = [list(tokenize(doc)) for doc in corpus]
    texts = TextCollection(tokenized_corpus)

    for doc in tokenized_corpus:
        yield {
            term: texts.tf_idf(term, doc)
            for term in doc
        }


for toks in nltk_tfidf_vectorize(paragraphs):
    print(toks)
"""


def sklearn_tfidf_vectorize(corpus):
    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(corpus)


df = pd.DataFrame(papers_list, columns={'title', 'abstract'})

tf_idf = sklearn_tfidf_vectorize(abstract_paragraphs)
tf_idf_df = pd.DataFrame(tf_idf.todense())

pca = PCA(n_components=2048) # 주성분을 몇개로 할지 결정
principalComponents = pca.fit_transform(tf_idf_df)
principalDf = pd.DataFrame(data=principalComponents)

fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(1, 1, 1, projection='3d')
#ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_zlabel('Principal Component 3', fontsize=15)
ax.set_title('3 component PCA', fontsize=20)

ax.scatter(principalDf[0],
           principalDf[1],
           principalDf[2])
plt.show()


DR_eps = 0.5
DR_min_samples = 2
model_DBSCAN = DBSCAN(eps=DR_eps, min_samples=DR_min_samples)
clusters_DR = model_DBSCAN.fit(principalDf)
result_DBSCAN_DR = model_DBSCAN.fit_predict(principalDf)
df['cluster_DR'] = result_DBSCAN_DR

score_samples_DR = silhouette_samples(principalComponents, df['cluster_DR'])
df['silhouette_coeff_DR'] = score_samples_DR
silhouette_s_DR = silhouette_score(principalComponents, df['cluster_DR'])

temp = 0
for e in df.groupby('cluster_DR')['silhouette_coeff_DR'].mean():
    temp += e
average_score_DR = temp/len(set(clusters_DR.labels_))
#average_score_DR_without_noise
print(type(df.groupby('cluster_DR')['silhouette_coeff_DR'].mean()))
print(df.groupby('cluster_DR')['silhouette_coeff_DR'].mean()[0])
print("average_score DR: ", average_score_DR)
print("silhouette score: ", silhouette_s_DR)
print(df.groupby('cluster_DR')['silhouette_coeff_DR'].mean())
print("DR eps, DR min_samples: ", DR_eps, DR_min_samples, "\n")

