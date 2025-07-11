# IMPORTATION 

import sklearn
import numpy as np
import matplotlib.pyplot as plt

# DATASET

iris = sklearn.datasets.load_iris()

features = iris.data
labels = iris.target
names = iris.target_names

#print(features)
#print(labels)
#print(iris.feature_names)
#print(iris.target_names)

# VISUALIZATION OF DATA

plt.scatter(features[:,0], features[:,1], c=labels)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.savefig('sep_length_vs_width.png')
plt.close()

plt.scatter(features[:,2], features[:,3], c=labels)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.savefig('pet_length_vs_width.png')
plt.close()

plt.scatter(features[:,0], features[:,2], c=labels)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.savefig('sep_length_vs_pet.png')
plt.close()

plt.scatter(features[:,1], features[:,3], c=labels)
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')
plt.savefig('sep_width_vs_pet.png')
plt.close()

# DIMENSION REDUCTION

"""
Since we have four numpy arrays for our iris we want to condense this data so that we don't analyze plots separately.
For my first time using data reduction I'm gonna use PCA.
It's all the more useful as we can for instance see that petal length and width seem to be strongly correlated therefore we can have one less dimension.
"""

pca = sklearn.decomposition.PCA(n_components=3)
pca.fit(features) # Get Information to prepare transformation
reduced_features = pca.transform(features) # Transform data to 3 components
#print(pca.explained_variance_ratio_) # Useful to know how much variance is carried by each newly created parameter

ax = plt.axes(projection="3d")
ax.scatter(reduced_features[:,0], reduced_features[:,1], reduced_features[:,2], c=labels)
ax.set_title('Reduced Plot')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.savefig('3D_reduced_plot.png')
plt.close()

"""
Let's try now to classify one new made up point by just looking to which centroid it is closer.
"""

fake_point = [[5, 3, 1.5, 0.3]]
point_reduction = pca.transform(fake_point)

def custom_dist(p1, p2, weights):
    """
    We need to calculate the barycenter and the distance of the newly created point to the barycenter to classify it.
    Yet since the variance is different along different axis I thought that I'd better use a custom distance.
    """
    return np.sum(weights*np.abs(p1-p2))

centroids = {}
for species in np.unique(labels):
    species_mask = (labels==species) # This is somehow possible because of numpy
    centroids[species] = np.mean(reduced_features[species_mask], axis=0)
     
distance_list = [custom_dist(list(centroids.values())[i], point_reduction[0], pca.explained_variance_ratio_) for i in range(len(centroids.values()))]

#print(f'The species type is {names[distance_list.index(min(distance_list))]}')


# CLASSIFICATION

"""
The dataset is prepared so we don't apply any transformation to it and take it raw.
We are going to start the classification here with the reduced data for visualization sake.
Maybe Later we will try classifying based on the real dataset.

Remark : the dataset is sorted so that's why we have to take random values I tested without sorting and obviously the less neighbors we look at the better the results.
"""


"""
First we will use a simple kNN algorithm because it suits the small dataset case well.
"""

from sklearn.neighbors import KNeighborsClassifier
train_ratio = 0.8
l = len(reduced_features)

def calculate_accuracy_NN(k):
    """
    Averages accuracy over several permutations of the dataset to assess the importance of the parameter k
    """
    
    list_accuracies = []

    for _ in range(1000):
        indices = np.random.permutation(l)

        features_permuted = reduced_features[indices]
        labels_permuted = labels[indices]

        features_train = features_permuted[:int(train_ratio*l)] 
        features_test = features_permuted[int(train_ratio*l):]

        labels_train = labels_permuted[:int(train_ratio*l)]
        labels_test = labels_permuted[int(train_ratio*l):]

        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(features_train,labels_train)

        predicted_labels = neigh.predict(features_test)
        true_labels = (predicted_labels==labels_test)

        list_accuracies.append(true_labels.sum()/len(predicted_labels)*100)
        
    print(f"The accuracy of the {k}-NN algorithm is on average : {np.mean(list_accuracies)}")

calculate_accuracy_NN(3)
calculate_accuracy_NN(4)
calculate_accuracy_NN(5)