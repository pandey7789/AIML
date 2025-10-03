import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


np.random.seed(42)
locations = ['LocA','LocB','LocC','LocD','LocE','LocF','LocG','LocH','LocI','LocJ']
ecological_data = {
    'Temperature': np.random.randint(15, 35, size=10),
    'Rainfall': np.random.randint(800, 2000, size=10),
    'Elevation': np.random.randint(100, 2000, size=10)
}


zone_labels = ['Zone1','Zone1','Zone2','Zone2','Zone2','Zone3','Zone3','Zone3','Zone1','Zone2']


df = pd.DataFrame(ecological_data, index=locations)
df['KnownZone'] = zone_labels


kmeans = KMeans(n_clusters=3, random_state=42)
df['AssignedCluster'] = kmeans.fit_predict(df[['Temperature', 'Rainfall', 'Elevation']])


print("Cluster Assignments vs Known Ecological Zones:")
print(df[['KnownZone', 'AssignedCluster']].sort_values('AssignedCluster'))


print("\nEstimated Cluster Purity:")
for cluster_id in sorted(df['AssignedCluster'].unique()):
    cluster_group = df[df['AssignedCluster'] == cluster_id]
    dominant_zone = cluster_group['KnownZone'].mode()[0]
    dominant_count = (cluster_group['KnownZone'] == dominant_zone).sum()
    total_in_cluster = len(cluster_group)
    purity_score = dominant_count / total_in_cluster
    print(f"Cluster {cluster_id}: Dominant Zone = {dominant_zone}, Purity = {purity_score:.2f}")
