import os
import bs4
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
import nltk
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.feature_extraction import text
from nltk import sent_tokenize
from nltk import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from numpy.linalg import svd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
#1시간정도걸림

#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('omw-1.4')

CAT_PATTERN = r'([a-z_\s]+)/.*'
DOC_PATTERN = r'(?!\.)[a-z_\s]+/[0-9]+\.html'
TAGS = []
title_TAGS = ['h1']
abstract_TAGS = ['blockquote']


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


def paras(htmls, TAGS):
    for html in htmls:
        soup = bs4.BeautifulSoup(html, 'lxml')
        for element in soup.find_all(TAGS):
            yield element.text
        soup.decompose()


def sents(paragraph):
    for sentence in sent_tokenize(paragraph):
        yield sentence


def words(paragraph):
    for sentence in sents(paragraph):
        for word in wordpunct_tokenize(sentence):
            yield word


def sklearn_tfidf_vectorize(corpus):
    my_stop_words = text.ENGLISH_STOP_WORDS.union(['abstract', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                                                   'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                                                   'w', 'x', 'y', 'z', ''])
    tfidf = TfidfVectorizer(token_pattern='[a-zA-Z]*', stop_words=my_stop_words, min_df=5, sublinear_tf=True, max_df=0.85)
    res = tfidf.fit_transform(corpus)
    print(tfidf.get_feature_names()[:100])
    return res


title_corpus = HTMLCorpusReader('', CAT_PATTERN, DOC_PATTERN, tags=title_TAGS)
title_fileids = title_corpus.fileids()
title_documents = title_corpus.docs(categories=title_corpus.categories())
title_htmls = list(title_documents)

abstract_corpus = HTMLCorpusReader('', CAT_PATTERN, DOC_PATTERN, abstract_TAGS)
abstract_fileids = abstract_corpus.fileids()
abstract_documents = abstract_corpus.docs(categories=abstract_corpus.categories())
abstract_htmls = list(abstract_documents)

title_categories = title_corpus.categories()
abstract_categories = abstract_corpus.categories()


title_paragraphs = list(paras(title_htmls, title_TAGS))
temp_title_paragraphs = []
for para in title_paragraphs:
    if "Title:" in para: # and len(para)>30
        temp_title_paragraphs.append(para.strip('Title:\n'))
title_paragraphs = temp_title_paragraphs
print("title_paragraphs len: ", len(title_paragraphs))
print("descreibe title_paragraphs", describe(title_paragraphs, title_fileids, title_categories))

abstract_paragraphs = list(paras(abstract_htmls, abstract_TAGS))
print("abstract_paragraphs len: ", len(abstract_paragraphs))
print("descreibe abstract_paragraphs", describe(abstract_paragraphs, abstract_fileids, abstract_categories))

papers_list = []
for key, value in zip(title_paragraphs, abstract_paragraphs):
    temp_dict = dict()
    temp_dict['title'] = key
    temp_dict['abstract'] = value
    papers_list.append(temp_dict)

print("papers_list[:5] : ", papers_list[:5])

print("title paragraphs[:5] : ", title_paragraphs[:5])

sparse_title_tf_idf = sklearn_tfidf_vectorize(title_paragraphs)
print(sparse_title_tf_idf)
print("type of sparse_title_tf_idf : ", type(sparse_title_tf_idf))

title_tf_idf = sparse_title_tf_idf.todense()
print(title_tf_idf)
print("type of title_tf_idf : ", type(title_tf_idf))

title_tf_idf_df = pd.DataFrame(title_tf_idf)
print(title_tf_idf_df.head())
print("title_tf_idf_df.shape : ", title_tf_idf_df.shape)

df = pd.DataFrame(papers_list, columns={'title', 'abstract'})
print(df.head())
print("df.shape : ", df.shape)
print("abstract_paragraphs: ", abstract_paragraphs[:5])

abstract_words = []
for p in abstract_paragraphs:
    abstract_words.append(list(words(p)))
print(abstract_words[0])
lemma_abstract_words = []
n=WordNetLemmatizer()
for words in abstract_words:
    lemma_abstract_words.append([n.lemmatize(w) for w in words])
print(lemma_abstract_words[0])
print(len(lemma_abstract_words))

lemma_abstract_paragraphs = []
for l in lemma_abstract_words:
    lemma_abstract_paragraphs.append(" ".join(l))
print(lemma_abstract_paragraphs[0])
print(len(lemma_abstract_paragraphs))

lemma_sparse_tf_idf = sklearn_tfidf_vectorize(lemma_abstract_paragraphs)
print(lemma_sparse_tf_idf)
print("type of lemma_sparse_tf_idf : ", type(lemma_sparse_tf_idf))

lemma_tf_idf = lemma_sparse_tf_idf.todense()
print(lemma_tf_idf)
print("type of lemma_tf_idf : ", type(lemma_tf_idf))

lemma_tf_idf_df = pd.DataFrame(lemma_tf_idf)
print(lemma_tf_idf_df.head())
print("lemma_tf_idf_df.shape : ", lemma_tf_idf_df.shape)

sparse_tf_idf = sklearn_tfidf_vectorize(abstract_paragraphs)
print(sparse_tf_idf)
print("type of sparse_tf_idf : ", type(sparse_tf_idf))

tf_idf = sparse_tf_idf.todense()
print(tf_idf)
print("type of tf_idf : ", type(tf_idf))

tf_idf_df = pd.DataFrame(tf_idf)
print(tf_idf_df.head())
print("tf_idf_df.shape : ", tf_idf_df.shape)

#title_tf_idf_df.to_csv('title_tf_idf_df.csv')
#df.to_csv('df.csv')
#tf_idf_df.to_csv('tf_idf_df.csv')
#lemma_tf_idf_df.to_csv('lemma_tf_idf_df.csv')
#print("csv saving done")

#pca = PCA().fit(tf_idf_df)
#
#plt.rcParams["figure.figsize"] = (12,6)
#
#fig, ax = plt.subplots()
#xi = np.arange(1, 13685, step=1)
#y = np.cumsum(pca.explained_variance_ratio_)
#
#plt.ylim(0.0,1.1)
#plt.plot(xi, y, marker='o', linestyle='--', color='b')
#
#plt.xlabel('Number of Components')
#plt.xticks(np.arange(0, 13685, step=1)) #change from 0-based array index to 1-based human-readable label
#plt.ylabel('Cumulative variance (%)')
#plt.title('The number of components needed to explain variance')
#
#plt.axhline(y=0.95, color='r', linestyle='-')
#plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)
#
#ax.grid(axis='x')
#plt.show()





#u, s, vt = svd(tf_idf)
#print("svd done")
#s = np.square(s)
#s = np.diag(s)
#s_list = []
#trace = np.trace(s)
#for i in range(0, 3450):
#    s_list.append(s[i][i]/trace)
#print("making list done")
#
#for i in range(1, 3450):
#    print(reduce(lambda a, b: a + b, s_list[:i])) #0.8