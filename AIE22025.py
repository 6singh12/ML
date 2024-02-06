#1
import math

def euclidean_distance(vector1, vector2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(vector1, vector2)))

def manhattan_distance(vector1, vector2):
    return sum(abs(x - y) for x, y in zip(vector1, vector2))


#2
from collections import Counter

def knn_classifier(train_data, test_instance, k):
    distances = [(euclidean_distance(train_instance, test_instance), label) for train_instance, label in train_data]
    sorted_distances = sorted(distances, key=lambda x: x[0])
    k_nearest_labels = [label for distance, label in sorted_distances[:k]]
    most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common_label


#3
def label_encoding(categories, input_data):
    label_mapping = {category: index for index, category in enumerate(categories)}
    encoded_data = [label_mapping[item] for item in input_data]
    return encoded_data


#4
def one_hot_encoding(categories, input_data):
    encoded_data = []
    for item in input_data:
        encoding = [0] * len(categories)
        encoding[categories.index(item)] = 1
        encoded_data.append(encoding)
    return encoded_data
