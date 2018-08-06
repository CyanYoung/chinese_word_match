import pandas as pd

from match import predict


def test(path_test, metric):
    errors = list()
    count = 0
    for text, label in pd.read_csv(path_test).values:
        pred = predict(text, metric)
        if pred != label:
            errors.append((text, label, pred))
        count = count + 1
    print('%s %.2f\n' % ('acc:', (count - len(errors)) / count))
    for text, label, pred in errors:
        print('{}, {} -> {}'.format(text, label, pred))


if __name__ == '__main__':
    path_test = 'data/test.csv'
    test(path_test, 'edit_dist')
    test(path_test, 'cos_sim')
