"""
Пространственная кластеризация
https://cartetika.ru/tpost/uun5jy5tk1-klasterizatsiya-prostranstvennih-dannih
"""
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

def spat_clust01(xmain): # , xcoo, ycoo, lbl
    k_means_df = pd.DataFrame(columns =
                              ['k', 'davies_bouldin_score', 'silhouette_score','calinski_harabasz_score','wcss_score'])
    for k in range(2, len(xmain)//2+1):
        model = KMeans(n_clusters=k, max_iter=300, init='k-means++', random_state=0).fit(xmain)
        class_predictions = model.predict(xmain)
        metric_silhouette_score = silhouette_score(xmain, class_predictions)
        metric_davies_bouldin_score = davies_bouldin_score(xmain, class_predictions)
        metric_calinski_harabasz_score = calinski_harabasz_score(xmain, class_predictions)
        metric_wcss_score = model.inertia_
        k_means_df.loc[len(k_means_df)] = {'k': k,
                                            'calinski_harabasz_score': metric_calinski_harabasz_score,
                                            'davies_bouldin_score': metric_davies_bouldin_score,
                                            'silhouette_score':metric_silhouette_score,
                                            'wcss_score':metric_wcss_score}
        # k_means_df = k_means_df.append({'k': k,
        #                                 'calinski_harabasz_score': metric_calinski_harabasz_score,
        #                                 'davies_bouldin_score': metric_davies_bouldin_score,
        #                                 'silhouette_score':metric_silhouette_score,
        #                                 'wcss_score':metric_wcss_score},ignore_index=True)

    k_means_df = k_means_df.sort_values(by=['silhouette_score','calinski_harabasz_score',
                                            'davies_bouldin_score', 'wcss_score'],
                                            ascending = [False, False, True, True])
    k_best = k_means_df.k.head().to_list()
    k_means_df = k_means_df.sort_values(by = 'k')
    print(k_best, type(k_best))

print('\npspat_clust01')