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
from scipy.sparse.linalg import svds


df = pd.read_csv('df.csv', index_col=0)
print(df.head())
print("df.shape : ", df.shape)

tf_idf_df = pd.read_csv('mindf005_tf_idf_df.csv', index_col=0)
print(tf_idf_df.head())
print("tf_idf_df.shape : ", tf_idf_df.shape)

pca = PCA().fit(tf_idf_df)

plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(1, tf_idf_df.shape[1]+1, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, tf_idf_df.shape[1]+1, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.axhline(y=0.90, color='r', linestyle='-')
plt.axhline(y=0.80, color='r', linestyle='-')
plt.axhline(y=0.5, color='r', linestyle='-')
plt.axhline(y=0.1, color='r', linestyle='-')
plt.text(0.5, 0.96, '95% cut-off threshold', color='red', fontsize=8)
plt.text(0.5, 0.91, '90% cut-off threshold', color='red', fontsize=8)
plt.text(0.5, 0.81, '80% cut-off threshold', color='red', fontsize=8)
plt.text(0.5, 0.51, '50% cut-off threshold', color='red', fontsize=8)
plt.text(0.5, 0.11, '10% cut-off threshold', color='red', fontsize=8)

ax.grid(axis='x')
plt.show()


def clustering(df_, tf_idf_df_, tf_idf_, eps_, min_samples_):
    if len(eps_)*len(min_samples_) == 1:
        ncols = 2
    else:
        ncols = len(eps_)

    fig, axs = plt.subplots(figsize=(8 * 1, 8), nrows=1, ncols=ncols) #change col
    ind = 0
    result = []
    average_score = []
    for i in range(len(eps_)):
        print("e,s: ", eps_[i], min_samples_[i],'\n')
        model = DBSCAN(eps=eps_[i], min_samples=min_samples_[i])
        clusters = model.fit(tf_idf_df_)
        n_cluster = len(set(clusters.labels_))
        if n_cluster <= 2:
            print("cluster num of", eps[i], min_samples_[i], "is 2 or less\n")
            continue
        #result.append(model.fit_predict(tf_idf_df_))
        result.append(clusters.labels_)
        df_['cluster' + 'of' + str(eps_[i]) + 'and' + str(min_samples_[i])] = result[ind]
        score_samples = silhouette_samples(tf_idf_, df_['cluster' + 'of' + str(eps_[i]) + 'and' + str(min_samples_[i])])
        df_['silhouette_coeff' + 'of' + str(eps_[i]) + 'and' + str(min_samples_[i])] = score_samples
        silhouette_s = silhouette_score(tf_idf_, df_['cluster' + 'of' + str(eps_[i]) + 'and' + str(min_samples_[i])])
        temp = 0
        for p in df_.groupby('cluster' + 'of' + str(eps_[i]) + 'and' + str(min_samples_[i]))['silhouette_coeff' + 'of' + str(eps_[i]) + 'and' + str(min_samples_[i])].mean():
            temp += p
        average_score.append(temp/len(set(clusters.labels_)))

        y_lower = 10
        axs[i].set_title(
            'Number of Cluster : ' + str(n_cluster) + '\n' + 'Silhouette Score :' + str(round(silhouette_s, 3)))
        axs[i].set_xlabel("The silhouette coefficient values")
        axs[i].set_ylabel("Cluster label")
        axs[i].set_xlim([-0.1, 1])
        axs[i].set_ylim([0, len(tf_idf_df_) + (n_cluster + 1) * 10])
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

    return result, df_, average_score


n_components = int(input('enter n_components : '))
U_tr, Sigma_tr, Vt_tr = svds(tf_idf_df, k=n_components)
principalComponents = U_tr
principalDf = pd.DataFrame(data=U_tr)
print("principalDf.shape : ", principalDf.shape)

#eps choose
n_neighbors = 2
neighbors = NearestNeighbors(n_neighbors=n_neighbors)
neighbors_fit = neighbors.fit(principalDf)
distances, indices = neighbors_fit.kneighbors(principalDf)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
#distances = np.sum(distances, axis=1)/(n_neighbors-1)
plt.plot(distances)
plt.show()

eps = input('enter eps: ')
eps = list(map(float, eps.split()))
print("eps: ", eps)
min_samples = [2 for i in range(len(eps))]

res, df, avg = clustering(df, principalDf, principalComponents, eps, min_samples)

print("average_score: ", avg)

print(res[0])
print(type(res[0]))
for cluster_num in set(res[0]):
     if cluster_num == -1 or cluster_num == 0:
         continue
     print("cluster num : {}".format(cluster_num))
     temp_df = df[df['cluster' + 'of' + str(eps[0]) + 'and' + str(min_samples[0])] == cluster_num] # cluster num 별로 조회
     for title in temp_df['title']:
         print(title) # 제목으로 살펴보자
         print()

