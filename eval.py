import pandas as pd

from sklearn.metrics import accuracy_score

from match import predict


def test(path_test, metric):
    labels = list()
    preds = list()
    errors = list()
    for text, label in pd.read_csv(path_test).values:
        labels.append(label)
        pred = predict(text, metric)
        preds.append(pred)
        if pred != label:
            errors.append((text, label, pred))
    print('%s %.2f\n' % ('acc:', accuracy_score(labels, preds)))
    for text, label, pred in errors:
        print('{}: {} -> {}'.format(text, label, pred))


if __name__ == '__main__':
    path_test = 'data/test.csv'
    test(path_test, 'edit_dist')
    test(path_test, 'cos_sim')
