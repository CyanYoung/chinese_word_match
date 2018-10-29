import pandas as pd

from sklearn.metrics import accuracy_score

from match import predict


def test(name, path):
    labels = list()
    preds = list()
    errors = list()
    for text, label in pd.read_csv(path).values:
        labels.append(label)
        pred = predict(text, name)
        preds.append(pred)
        if pred != label:
            errors.append((text, label, pred))
    print('%s %.2f\n' % ('acc:', accuracy_score(labels, preds)))
    for text, label, pred in errors:
        print('{}: {} -> {}'.format(text, label, pred))


if __name__ == '__main__':
    path = 'data/test.csv'
    test('edit', path)
    test('cos', path)
