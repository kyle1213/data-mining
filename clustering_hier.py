import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from functools import reduce
from numpy.linalg import svd
from sklearn.cluster import AgglomerativeClustering


df = pd.read_csv('df.csv', index_col=0)
print(df.head())
print("df.shape : ", df.shape)

tf_idf_df = pd.read_csv('title_tf_idf_df.csv', index_col=0)
print(tf_idf_df.head())
print("tf_idf_df.shape : ", tf_idf_df.shape)


def clustering(df_, tf_idf_df_, tf_idf_):
    fig, axs = plt.subplots(figsize=(8 * 1, 8), nrows=1, ncols=1)
    model = AgglomerativeClustering(linkage="ward",
                                    distance_threshold=None,
                                    n_clusters=128)

    clusters = model.fit(tf_idf_df_)
    n_cluster = len(set(clusters.labels_))
    result = model.fit_predict(tf_idf_df_)
    df_['cluster'] = result
    score_samples = silhouette_samples(tf_idf_df_, df_['cluster'])
    df_['silhouette_coeff'] = score_samples
    silhouette_s = silhouette_score(tf_idf_df_, df_['cluster'])
    temp = 0
    for p in df_.groupby('cluster')['silhouette_coeff'].mean():
        temp += p
    average_score = temp/len(set(clusters.labels_))

    y_lower = 10
    axs.set_title(
        'Number of Cluster : ' + str(n_cluster) + '\n' + 'Silhouette Score :' + str(round(silhouette_s, 3)))
    axs.set_xlabel("The silhouette coefficient values")
    axs.set_ylabel("Cluster label")
    axs.set_xlim([-0.1, 1])
    axs.set_ylim([0, len(tf_idf_df_) + (n_cluster + 1) * 10])
    axs.set_yticks([])  # Clear the yaxis labels / ticks
    axs.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현.
    for j in range(-1, n_cluster-1):
        ith_cluster_sil_values = score_samples[result == j]
        ith_cluster_sil_values.sort()

        size_cluster_i = ith_cluster_sil_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(j) / n_cluster)
        axs.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                               facecolor=color, edgecolor=color, alpha=0.7)
        axs.text(-0.05, y_lower + 0.5 * size_cluster_i, str(j))
        y_lower = y_upper + 10

    axs.axvline(x=silhouette_s, color="red", linestyle="--")
    plt.show()

    return result


pca = PCA(n_components=0.8) # 기여율을 위해 feature수(or sample수)(둘 중 작은 수) 만큼 pca해주기 /
principalComponents = pca.fit_transform(tf_idf_df)
principalDf = pd.DataFrame(data=principalComponents)
print("principalDf.shape : ", principalDf.shape)

res = clustering(df, principalDf, principalComponents)

print(res[0])
print(type(res[0]))
