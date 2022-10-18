import os
import bs4
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
import nltk
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.neighbors import NearestNeighbors
import re
from sklearn.feature_extraction import text
from sklearn.decomposition import PCA


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
print("descreibe title_paragraphs", describe(title_paragraphs, title_fileids, title_categories))

abstract_paragraphs = list(paras(abstract_htmls, abstract_TAGS))
print("abstract_paragraphs len: ", len(abstract_paragraphs))
print("descreibe abstract_paragraphs", describe(abstract_paragraphs, abstract_fileids, abstract_categories))

#temp_para = []
#print(abstract_paragraphs[0])
#for para in abstract_paragraphs:
#    temp_para.append(re.sub(r"[^a-zA-Z\s.]", "", para).lower())  # 영문자 + 공백만 남기기)
#abstract_paragraphs = temp_para
#print("descreibe post abstract_paragraphs", describe(abstract_paragraphs, abstract_fileids, abstract_categories))
#print(abstract_paragraphs[0])


papers_list = []
for key, value in zip(title_paragraphs, abstract_paragraphs):
    temp_dict = dict()
    temp_dict['title'] = key
    temp_dict['abstract'] = value
    papers_list.append(temp_dict)


def sklearn_tfidf_vectorize(corpus):
    my_stop_words = text.ENGLISH_STOP_WORDS.union(['abstract', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                                                   'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                                                   'w', 'x', 'y', 'z'])
    tfidf = TfidfVectorizer(stop_words=my_stop_words)
    return tfidf.fit_transform(corpus)


def clustering(df, tf_idf_df, tf_idf, eps, min_samples):
    neighbors = NearestNeighbors(n_neighbors=4)
    neighbors_fit = neighbors.fit(tf_idf_df)
    distances, indices = neighbors_fit.kneighbors(tf_idf_df)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()

    if len(eps)*len(min_samples) == 1:
        ncols = 2
    else:
        ncols = len(eps)

    fig, axs = plt.subplots(figsize=(8 * 1, 8), nrows=1, ncols=ncols) #change col
    ind = 0
    result = []
    for i in range(len(eps)):
        print("e,s: ", eps[i], min_samples[i],'\n')
        model = DBSCAN(eps=eps[i], min_samples=min_samples[i])
        clusters = model.fit(tf_idf_df)
        n_cluster = len(set(clusters.labels_))
        if n_cluster <= 2:
            print("cluster num of", eps[i], min_samples[i], "is 2 or less\n")
            continue
        result.append(model.fit_predict(tf_idf_df))
        df['cluster' + 'of' + str(eps[i]) + 'and' + str(min_samples[i])] = result[ind]
        score_samples = silhouette_samples(tf_idf, df['cluster' + 'of' + str(eps[i]) + 'and' + str(min_samples[i])])
        df['silhouette_coeff' + 'of' + str(eps[i]) + 'and' + str(min_samples[i])] = score_samples
        silhouette_s = silhouette_score(tf_idf, df['cluster' + 'of' + str(eps[i]) + 'and' + str(min_samples[i])])
        temp = 0
        for p in df.groupby('cluster' + 'of' + str(eps[i]) + 'and' + str(min_samples[i]))['silhouette_coeff' + 'of' + str(eps[i]) + 'and' + str(min_samples[i])].mean():
            temp += p
        average_score = temp/len(set(clusters.labels_))

        y_lower = 10
        axs[i].set_title(
            'Number of Cluster : ' + str(n_cluster) + '\n' + 'Silhouette Score :' + str(round(silhouette_s, 3)))
        axs[i].set_xlabel("The silhouette coefficient values")
        axs[i].set_ylabel("Cluster label")
        axs[i].set_xlim([-0.1, 1])
        axs[i].set_ylim([0, len(tf_idf_df) + (n_cluster + 1) * 10])
        axs[i].set_yticks([])  # Clear the yaxis labels / ticks
        axs[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현.
        for j in range(-1, n_cluster-1):
            ith_cluster_sil_values = score_samples[result[ind] == j]
            ith_cluster_sil_values.sort()

            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(j) / n_cluster)
            axs[i].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                   facecolor=color, edgecolor=color, alpha=0.7)
            axs[i].text(-0.05, y_lower + 0.5 * size_cluster_i, str(j))
            y_lower = y_upper + 10

        axs[i].axvline(x=silhouette_s, color="red", linestyle="--")
        ind += 1
    plt.show()

    return result


dff = pd.DataFrame(papers_list, columns={'title', 'abstract'})
tf_idff = sklearn_tfidf_vectorize(abstract_paragraphs).todense()
tf_idf_dff = pd.DataFrame(tf_idff)

pca = PCA(n_components=18) # 주성분을 몇개로 할지 결정
principalComponents = pca.fit_transform(tf_idf_dff)
principalDf = pd.DataFrame(data=principalComponents)

eps = [0.05, 0.03, 0.09, 0.05, 0.03, 0.09, 0.05, 0.03, 0.09]
min_samples = [2, 2, 2, 4, 4, 4, 7, 7, 7]
res = clustering(dff, principalDf, principalComponents, eps, min_samples)

for i, r in enumerate(res):
    if set(r) == None:
        continue
    else:
        for cluster_num in set(r):
            if cluster_num == -1 or cluster_num == 0:
                continue
        # -1,0은 노이즈 판별이 났거나 클러스터링이 안된 경우
            print("cluster num : {}".format(cluster_num))
            temp_df = dff[dff['cluster' + 'of' + str(eps[i]) + 'and' + str(min_samples[i])] == cluster_num] # cluster num 별로 조회
            for title in temp_df['title']:
                print(title) # 제목으로 살펴보자
                print()

    print("-----------------\n")
"""

        print(df.head())
        print("num of clusters: ", len(set(clusters.labels_)))
        print("average_score: " + 'of' + str(e) + 'and' + str(s) + ": ", average_score)
        print("silhouette score: " + 'of' + str(e) + 'and' + str(s) + ": ", silhouette_s)
        print(df.groupby('cluster' + 'of' + str(e) + 'and' + str(s))['silhouette_coeff' + 'of' + str(e) + 'and' + str(s)].mean())
        print("eps, min_samples: " + 'of' + str(e) + 'and' + str(s) + ": ", e, s)"""