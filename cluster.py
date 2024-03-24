from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import csv
import json


def main():
    csv_file = open('data/question_subject_meta_clean.csv', 'r')
    csv_reader = csv.reader(csv_file)
    # Skip the header
    next(csv_reader)
    # Convert the file into a list of lists where each item is of the form [int, [str, str, ...]]]
    data = [[int(row[0]), json.loads(row[1].replace("'", "\""))] for row in csv_reader]
    csv_file.close()

    # Extract the subjects
    subjects = [", ".join(item[1]) for item in data]

    # Use the TfidfVectorizer to convert the subjects into a matrix
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(subjects)

    # Use KMeans to cluster the subjects
    clust = KMeans(n_clusters=100, random_state=0)
    clust.fit(X)

    # Write the clusters to a file
    with open('data/cluster.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['question_id', 'cluster'])
        for i, item in enumerate(data):
            writer.writerow([item[0], clust.labels_[i]])


if __name__ == "__main__":
    main()
