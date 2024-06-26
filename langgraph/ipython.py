import utils

def runCode():
    codeString = """
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.datasets import make_blobs
        from sklearn.cluster import KMeans

        # Create synthetic data
        X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

        # Plot the synthetic data
        plt.scatter(X[:, 0], X[:, 1], s=50)
        plt.show()

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)

        # Plot the data with the clusters identified by K-means
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

        # Plot the cluster centers
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
        plt.show()
        """
    lines = codeString.split('\n')
    lines = list(map(lambda line: line.replace('        ', ''), lines))
    codeString = '\n'.join(lines)
    utils.pythonProcess(codeString)

if __name__ == '__main__':
    # have to use the if __name__ == '__main__' to launch a new process in utils.pythonProcess()
    runCode()