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
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.neighbors import NearestNeighbors


#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

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


title_corpus = HTMLCorpusReader('', CAT_PATTERN, DOC_PATTERN, tags=title_TAGS)
title_fileids = title_corpus.fileids()
title_documents = title_corpus.docs(categories=title_corpus.categories())
title_htmls = list(title_documents)


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

title_categories = title_corpus.categories()


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
#print("descreibe abstract_paragraphs", describe(abstract_paragraphs, abstract_fileids, abstract_categories))



def sklearn_tfidf_vectorize(corpus):
    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(corpus)


def clustering(df, tf_idf_df):
    eps = [0.9]
    #eps = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 10]
    min_samples = [4]

    neighbors = NearestNeighbors(n_neighbors=4)
    neighbors_fit = neighbors.fit(tf_idf_df)
    distances, indices = neighbors_fit.kneighbors(tf_idf_df)
    distances = np.sort(distances, axis=0)
    distances = np.sum(distances, axis=1)/4
    plt.plot(distances)
    plt.show()

    fig, axs = plt.subplots(figsize=(8 * 1, 8), nrows=1, ncols=len(eps)*len(min_samples)) #change col

    ind = 0
    for e in eps:
        for s in min_samples:
            print("e,s: ", e, s,'\n')
            model = DBSCAN(eps=e, min_samples=s)
            clusters = model.fit(tf_idf_df)
            n_cluster = len(set(clusters.labels_))
            if n_cluster <= 2:
                print("cluster num of", e, s, "is 2 or less\n")
                continue
            result = model.fit_predict(tf_idf_df)
            df['cluster' + 'of' + str(e) + 'and' + str(s)] = result
            score_samples = silhouette_samples(tf_idf, df['cluster' + 'of' + str(e) + 'and' + str(s)])
            df['silhouette_coeff' + 'of' + str(e) + 'and' + str(s)] = score_samples
            silhouette_s = silhouette_score(tf_idf, df['cluster' + 'of' + str(e) + 'and' + str(s)])
            temp = 0
            for p in df.groupby('cluster' + 'of' + str(e) + 'and' + str(s))['silhouette_coeff' + 'of' + str(e) + 'and' + str(s)].mean():
                temp += p
            average_score = temp/len(set(clusters.labels_))

            y_lower = 10
            axs[ind].set_title(
                'Number of Cluster : ' + str(n_cluster) + '\n' + 'Silhouette Score :' + str(round(silhouette_s, 3)))
            axs[ind].set_xlabel("The silhouette coefficient values")
            axs[ind].set_ylabel("Cluster label")
            axs[ind].set_xlim([-0.1, 1])
            axs[ind].set_ylim([0, len(tf_idf_df) + (n_cluster + 1) * 10])
            axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
            axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

            # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현.
            for i in range(0, n_cluster-1):
                ith_cluster_sil_values = score_samples[result == i]
                ith_cluster_sil_values.sort()

                size_cluster_i = ith_cluster_sil_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_cluster)
                axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                       facecolor=color, edgecolor=color, alpha=0.7)
                axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10

            axs[ind].axvline(x=silhouette_s, color="red", linestyle="--")
            ind += 1

    plt.show()


df = pd.DataFrame()
tf_idf = sklearn_tfidf_vectorize(title_paragraphs).todense()
tf_idf_df = pd.DataFrame(tf_idf)
clustering(df, tf_idf)
"""

        print(df.head())
        print("num of clusters: ", len(set(clusters.labels_)))
        print("average_score: " + 'of' + str(e) + 'and' + str(s) + ": ", average_score)
        print("silhouette score: " + 'of' + str(e) + 'and' + str(s) + ": ", silhouette_s)
        print(df.groupby('cluster' + 'of' + str(e) + 'and' + str(s))['silhouette_coeff' + 'of' + str(e) + 'and' + str(s)].mean())
        print("eps, min_samples: " + 'of' + str(e) + 'and' + str(s) + ": ", e, s)"""