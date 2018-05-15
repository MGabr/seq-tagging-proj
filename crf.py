from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score, flat_recall_score, flat_precision_score


def tuple2features(tuples, i):
    # features are the token and the POS tag of current, previous and next token
    features = {
        "token": tuples[i][0],
        "pos tag": tuples[i][1],
        "prev pos tag": tuples[i - 1][1] if i > 0 else ".",
        "next pos tag": tuples[i + 1][1] if (i + 1) < len(tuples) else "."
    }
    return features


def tuple2label(t):
    return t[2]


def file2features_labels(filename):
    # we require sets of tuples - set contains all tuples until an empty line is reached
    tuple_sets = [[]]
    for line in open(filename):
        if line == "\n":
            tuple_sets.append([])
        else:
            tuple_sets[-1].append(line.rstrip().split(" "))

    features = [[tuple2features(tuples, i) for i in range(0, len(tuples))] for tuples in tuple_sets]
    labels = [[tuple2label(t) for t in tuples] for tuples in tuple_sets]
    return features, labels


print("Training CRF model on training data...")
train_features, train_labels = file2features_labels("train.txt")
crf = CRF()
crf.fit(train_features, train_labels)

print("Making predictions for test data...")
test_features, test_labels = file2features_labels("test.txt")
test_preds = crf.predict(test_features)

print("Performing own evaluation...")
labels = crf.classes_
p = flat_precision_score(test_labels, test_preds, labels=labels, average="micro")
r = flat_recall_score(test_labels, test_preds, labels=labels, average="micro")
f1 = flat_f1_score(test_labels, test_preds, labels=labels, average="micro")
print("p(micro)={} r(micro)={} f1(micro)={}".format(p, r, f1))


def to_conllevalfile(features, labels, preds, filename):
    with open(filename, "w") as conlleval_input_file:
        for feature_set, label_set, pred_set in zip(features, labels, preds):
            for feature, label, pred in zip(feature_set, label_set, pred_set):
                conlleval_input_file.write("{} {} {} {}\n".format(
                    feature["token"], feature["pos tag"], label, pred))
            conlleval_input_file.write("\n")


to_conllevalfile(test_features, test_preds, test_labels, "conlleval_input_crf.txt")
