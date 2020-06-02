import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    start = time.time()
    train_f = "mnist_train.csv"
    test_f = "mnist_test.csv"
    train_data, validation_data, train_labels, validation_labels = [], [], [], []
    for line in open(train_f):
        label_features = [int(x) for x in line.split(",")]
        train_data.append(label_features[1:])
        train_labels.append(label_features[0])

    for line in open(test_f):
        label_features = [int(x) for x in line.split(",")]
        validation_data.append(label_features[1:])
        validation_labels.append(label_features[0])

    clf = KNeighborsClassifier(n_neighbors=100, algorithm='auto', weights='distance')
    clf.fit(train_data, train_labels)
    pred = clf.predict(validation_data)
    end = time.time()
    print("Elapsed Time %.2f seconds\n" % (end - start))
    print('Accurancy: %.2f%%\n' % (accuracy_score(validation_labels, pred) * 100))


if __name__ == "__main__":
    main()
