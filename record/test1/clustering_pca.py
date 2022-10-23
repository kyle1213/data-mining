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


df = pd.read_csv('df.csv', index_col=0)
print(df.head())
print("df.shape : ", df.shape)

tf_idf_df = pd.read_csv('lemma_tf_idf_df.csv', index_col=0)
print(tf_idf_df.head())
print("tf_idf_df.shape : ", tf_idf_df.shape)


def clustering(df_, tf_idf_df_, tf_idf_, eps_, min_samples_):
    if len(eps_)*len(min_samples_) == 1:
        ncols = 2
    else:
        ncols = len(eps_)

    fig, axs = plt.subplots(figsize=(8 * 1, 8), nrows=1, ncols=ncols) #change col
    ind = 0
    result = []
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
        average_score = temp/len(set(clusters.labels_))

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

    return result, df_


#n_comp = 4800(80%), 10610(95%)
#n = 3450, 7200 with smoothing, min_df=5
pca = PCA(n_components=0.8) # 기여율을 위해 feature수(or sample수)(둘 중 작은 수) 만큼 pca해주기 /
principalComponents = pca.fit_transform(tf_idf_df)
principalDf = pd.DataFrame(data=principalComponents)
print("principalDf.shape : ", principalDf.shape)

#eps choose
neighbors = NearestNeighbors(n_neighbors=10)
neighbors_fit = neighbors.fit(principalDf)
distances, indices = neighbors_fit.kneighbors(principalDf)
distances = np.sort(distances, axis=0)
distances = np.sum(distances, axis=1)/10
print(np.shape(distances))
plt.plot(distances)
plt.show()

eps = [0.95, 0.95, 1, 1, 1.2, 1.2]
min_samples = [2, 10, 2, 10, 2, 10]
res, df = clustering(df, principalDf, principalComponents, eps, min_samples)

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


"""
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
"""

        print(df.head())
        print("num of clusters: ", len(set(clusters.labels_)))
        print("average_score: " + 'of' + str(e) + 'and' + str(s) + ": ", average_score)
        print("silhouette score: " + 'of' + str(e) + 'and' + str(s) + ": ", silhouette_s)
        print(df.groupby('cluster' + 'of' + str(e) + 'and' + str(s))['silhouette_coeff' + 'of' + str(e) + 'and' + str(s)].mean())
        print("eps, min_samples: " + 'of' + str(e) + 'and' + str(s) + ": ", e, s)"""