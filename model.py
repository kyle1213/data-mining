import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from flask import Flask, render_template, request
app = Flask(__name__)


df = pd.read_csv('df.csv', index_col=0)

tf_idf_df = pd.read_csv('mindf001_tf_idf_df.csv', index_col=0)
eps = 0.945


def clustering(df_, tf_idf_df_, tf_idf_, eps_):
    average_score = []
    model = DBSCAN(eps=eps_, min_samples=2)
    clusters = model.fit(tf_idf_df_)
    result = clusters.labels_
    df_['cluster' + 'of' + str(eps_) + 'and' + str(2)] = result
    score_samples = silhouette_samples(tf_idf_, df_['cluster' + 'of' + str(eps_) + 'and' + str(2)])
    df_['silhouette_coeff' + 'of' + str(eps_) + 'and' + str(2)] = score_samples
    temp = 0
    for p in df_.groupby('cluster' + 'of' + str(eps_) + 'and' + str(2))['silhouette_coeff' + 'of' + str(eps_) + 'and' + str(2)].mean():
        temp += p
    average_score.append(temp/len(set(clusters.labels_)))
    return result, df_, average_score


pca = PCA(n_components=0.8)
principalComponents = pca.fit_transform(tf_idf_df)
principalDf = pd.DataFrame(data=principalComponents)

res, df, avg = clustering(df, principalDf, principalComponents, eps)
pd.set_option('display.max_columns', None)

clf = NearestCentroid()

clf.fit(principalDf.to_numpy(), df['cluster' + 'of' + str(eps) + 'and' + str(2)])

#Adversarial Attacks on Transformers-Based Malware Detectors
#test_abstract = ["Signature-based malware detectors have proven to be insufficient as even a small change in malignant executable code can bypass these signature-based detectors. Many machine learning-based models have been proposed to efficiently detect a wide variety of malware. Many of these models are found to be susceptible to adversarial attacks - attacks that work by generating intentionally designed inputs that can force these models to misclassify. Our work aims to explore vulnerabilities in the current state of the art malware detectors to adversarial attacks. We train a Transformers-based malware detector, carry out adversarial attacks resulting in a misclassification rate of 23.9% and propose defenses that reduce this misclassification rate to half. An implementation of our work can be found at this https URL"]
test_abstract = ["Unmanned Aerial Vehicles (UAVs) are increasingly deployed to provide wireless connectivity to static and mobile ground users in situations of increased network demand or points-of-failure in existing terrestrial cellular infrastructure. However, UAVs are energy-constrained and may experience interference from nearby UAV cells sharing the same frequency spectrum, thereby impacting the system's energy efficiency (EE). We aim to address research gaps that focus on optimising the system's EE using a 2D trajectory optimisation of UAVs serving only static ground users, and neglect the impact of interference from nearby UAV cells. Unlike previous work that assume global spatial knowledge of ground users' location via a central controller that periodically scans the network perimeter and provides real-time updates to the UAVs for decision making, we focus on a realistic decentralised approach suitable in emergencies. Thus, we apply a decentralised Multi-Agent Reinforcement Learning (MARL) approach that maximizes the system's EE by jointly optimising each UAV's 3D trajectory, number of connected static and mobile users, and the energy consumed, while taking into account the impact of interference and the UAVs' coordination on the system's EE in a dynamic network environment. To address this, we propose a direct collaborative Communication-enabled Multi-Agent Decentralised Double Deep Q-Network (CMAD-DDQN) approach. The CMAD-DDQN is a collaborative algorithm that allows UAVs to explicitly share knowledge by communicating with its nearest neighbours based on existing 3GPP guidelines. Our approach is able to maximise the system's EE without degrading the coverage performance in the network. Simulation results show that the proposed approach outperforms existing baselines in term of maximising the systems' EE by about 15% - 85%."]







def mindf001_sklearn_tfidf_vectorize(corpus, data):
    my_stop_words = text.ENGLISH_STOP_WORDS.union(['abstract', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                                                   'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                                                   'w', 'x', 'y', 'z', ''])
    tfidf = TfidfVectorizer(token_pattern='[a-zA-Z]*', stop_words=my_stop_words, min_df=0.01, sublinear_tf=True, max_df=0.85)
    tfidf.fit_transform(data)
    res = tfidf.transform(corpus)
    return res





@app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'POST':
        abs = request.form['abs']
        abs = [abs]
        data = df['abstract'].to_numpy()
        x = mindf001_sklearn_tfidf_vectorize(abs, data)

        x_tf_idf = x.todense()
        x_tf_idf_df = pd.DataFrame(x_tf_idf)
        x_principalComponents = pca.transform(x_tf_idf_df)
        x_principalDf = pd.DataFrame(data=x_principalComponents)
        a = clf.predict(x_principalDf.to_numpy().reshape(1, -1))[0]

        if a == -1:
            return "no matching cluster found"
        else:
            for cluster_num in set(res):
                if cluster_num == a:
                    temp_df = df[df['cluster' + 'of' + str(eps) + 'and' + str(2)] == cluster_num]  # cluster num 별로 조회
                    titles = " / ".join(temp_df['title'])
                    print(titles)
            return 'recommending paper titles: %s' %titles
    else:
        return render_template("main.html")


if __name__ == '__main__':
    app.run(debug=True, port=5000)