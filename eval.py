from sklearn.metrics import accuracy_score, f1_score

from match import predict

from util import flat_read


path_test = 'data/test.csv'
texts = flat_read(path_test, 'text')
labels = flat_read(path_test, 'label')


def test(name, texts, labels):
    preds, errors = list(), list()
    for text, label in zip(texts, labels):
        pred = predict(text, name)
        preds.append(pred)
        if pred != label:
            errors.append((text, label, pred))
    f1 = f1_score(labels, preds)
    print('\n%s f1: %.2f - acc: %.2f\n' % (name, f1, accuracy_score(labels, preds)))
    for text, label, pred in errors:
        print('{}: {} -> {}'.format(text, label, pred))


if __name__ == '__main__':
    test('edit', texts, labels)
    test('cos', texts, labels)
