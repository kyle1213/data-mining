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

temp_tf_idf = tf_idf.todense()*10
temp_tf_idf_df = tf_idf_df.mul(10)


pca = PCA(n_components=64) # 주성분을 몇개로 할지 결정
temp_principalComponents = pca.fit_transform(temp_tf_idf_df)
temp_principalDf = pd.DataFrame(data=temp_principalComponents)

fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(1, 1, 1, projection='3d')
#ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_zlabel('Principal Component 3', fontsize=15)
ax.set_title('3 component PCA', fontsize=20)

ax.scatter(temp_principalDf[0],
           temp_principalDf[1],
           temp_principalDf[2])
plt.show()


temp_eps = 0.5
temp_min_samples = 2
temp_model_DBSCAN = DBSCAN(eps=temp_eps, min_samples=temp_min_samples)
temp_clusters = temp_model_DBSCAN.fit(temp_tf_idf_df)
temp_result_DBSCAN = temp_model_DBSCAN.fit_predict(temp_tf_idf_df)
df['temp_cluster'] = temp_result_DBSCAN


temp_eps_DR = 0.5
temp_min_samples_DR = 2
temp_model_DBSCAN_DR = DBSCAN(eps=temp_eps_DR, min_samples=temp_min_samples_DR)
temp_clusters_DR = temp_model_DBSCAN_DR.fit(temp_principalDf)
temp_result_DBSCAN_DR = temp_model_DBSCAN_DR.fit_predict(temp_principalDf)
df['temp_cluster_DR'] = temp_result_DBSCAN_DR

#for cluster_num in set(result):
#    # -1,0은 노이즈 판별이 났거나 클러스터링이 안된 경우
#    print("cluster num : {}".format(cluster_num))
#    temp_df = df[df['cluster'] == cluster_num] # cluster num 별로 조회
#    for title in temp_df['title']:
#        print(title) # 제목으로 살펴보자
#        print()

temp_score_samples = silhouette_samples(temp_tf_idf, df['temp_cluster'])
df['temp_silhouette_coeff'] = temp_score_samples
temp_average_score = silhouette_score(temp_tf_idf, df['temp_cluster'])

print("temp_average_score: ", temp_average_score)
print("temp eps, min_samples: ", temp_eps, temp_min_samples, "\n")


temp_score_samples_DR = silhouette_samples(temp_principalComponents, df['temp_cluster_DR'])
df['temp_silhouette_coeff_DR'] = temp_score_samples_DR
temp_average_score_DR = silhouette_score(temp_principalComponents, df['temp_cluster_DR'])

print("temp_average_score_DR: ", temp_average_score_DR)
print("temp eps, min_samples: ", temp_eps_DR, temp_min_samples_DR, "\n")

# 할 수 있는거, 차원 축소하고 클러스터링하기
# or 지금 결과를 3차원으로 pca하여 시각화하기
# or 클러스터 최적의 파라미터 찾기(노이즈 줄이기)
# or 실루엣 계수같은 스코어링 확인해보기
