from flask import Flask
app = Flask(__name__)

@app.route('/')

def hello_world():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt 
    import seaborn as sns
    from sklearn.cluster import KMeans 
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import MinMaxScaler

    iris = pd.read_csv("./iris.csv")
    x = iris.iloc[:, [0, 1, 2, 3]].values
    iris.info()

    #Frequency distribution of species"
    iris_outcome = pd.crosstab(index=iris["variety"],  # Make a crosstab
                              columns="count")      # Name the count column

    iris_outcome

    iris_setosa=iris.loc[iris["variety"]=="Setosa"]
    iris_virginica=iris.loc[iris["variety"]=="Virginica"]
    iris_versicolor=iris.loc[iris["variety"]=="Versicolor"]
    sns.boxplot(x="variety",y="petal.length",data=iris)
    #plt.show()
    plt.savefig('Fig1.png')
    #Finding the optimum number of clusters for k-means classification
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    fig = plt.figure(figsize=(9, 7))
    plt.plot(range(1, 11), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS') #within cluster sum of squares
    #plt.show()
    plt.savefig('Fig2.png')
    kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    y_kmeans = kmeans.fit_predict(x)
    #Visualising the clusters
    fig = plt.figure(figsize=(9, 7))
    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Setosa')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Versicolour')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Virginica')

    #Plotting the centroids of the clusters
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
    plt.legend()
    plt.savefig('Kmeans.png') 
    

hello_world()
