import pandas as pd

from match import predict


def test(path):
    errors = list()
    count = 0
    for text, label in pd.read_csv(path).values:
        pred = predict(text)
        if pred != label:
            errors.append((text, label, pred))
        count = count + 1
    print('acc: {}\n'.format((count - len(errors)) / count))
    for text, label, pred in errors:
        print('{}, {} -> {}'.format(text, label, pred))


if __name__ == '__main__':
    path = 'data/test.csv'
    test(path)
