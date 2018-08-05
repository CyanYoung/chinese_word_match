import pandas as pd

from match import predict


def test(path, metric):
    errors = list()
    count = 0
    for text, label in pd.read_csv(path).values:
        pred = predict(text, metric)
        if pred != label:
            errors.append((text, label, pred))
        count = count + 1
    print('%s %.2f\n' % ('acc:', (count - len(errors)) / count))
    for text, label, pred in errors:
        print('{}, {} -> {}'.format(text, label, pred))


if __name__ == '__main__':
    path = 'data/test.csv'
    test(path, 'edit_dist')
    test(path, 'cos_sim')
