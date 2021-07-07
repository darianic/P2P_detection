import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from matplotlib import pyplot as plt


# удаление ненужных столбцов + добавление метки
def preprocess_data(df, class_label):
    # удаление ненужных столбцов
    df = df.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'timestamp'], axis=1)
    # удаление потоков со значением Nan в столбцах
    df = df.dropna()
    # добавление метки
    df['label'] = class_label
    return df


# создание смешанного датасета
def get_merged_df(df1, n1, df2, n2):
    df1 = df1.sample(n1)
    df2 = df2.sample(n2)
    mixed_df = pd.DataFrame(pd.concat((df1, df2), axis=0))
    mixed_df = mixed_df.sample(len(mixed_df))
    return mixed_df


def get_confusion_matrix(y_actual, y_true):
    y_actual = np.array(y_actual)
    y_true = np.array(y_true)
    TP = 0  # мера положительных записей, классифицированных верно
    FP = 0  # мера отрицательных записей неправильно классифицированных, как положительные
    TN = 0  # мера негативных записей, классифицированных верно
    FN = 0  # мера позитивных записей неправильно классифицированных, как отрицательные
    for i in range(len(y_true)):
        if y_actual[i] == y_true[i] == 1:
            TP += 1
        if y_true[i] == 1 and y_actual[i] != y_true[i]:
            FP += 1
        if y_actual[i] == y_true[i] == 0:
            TN += 1
        if y_true[i] == 0 and y_actual[i] != y_true[i]:
            FN += 1
    return [TN, FP, FN, TP]


def eval_model(model, X_test, y_test, model_name, verbose=1):
    res = model.predict(X_test)
    metrics = {}
    TN, FP, FN, TP = metrics['Confusion'] = get_confusion_matrix(res, y_test)
    metrics['Accuracy'] = (TP + TN) / np.sum(metrics['Confusion'])
    pr = metrics['Precision'] = TP / (TP + FP)
    rec = metrics['Recall'] = TP / (TP + FN)
    metrics['Specificity'] = TN / (FP + TN)
    metrics['F-score'] = 2 * (pr * rec) / (pr + rec)
    if verbose:
        from pprint import pprint
        print('Model {} training results: '.format(model_name))
        pprint(metrics)
    return metrics


p2p_df = pd.read_csv('p2p-12-04.csv', sep=',', low_memory=False)
p2p_df = preprocess_data(p2p_df, 1)
other_df = pd.read_csv('http_https_df.csv', sep=',', low_memory=False)
other_df = preprocess_data(other_df, 0)
mixed_df = get_merged_df(p2p_df, len(p2p_df), other_df, len(other_df))

sel = VarianceThreshold()
sel.fit(mixed_df.values)  # эмпирические отклонения от X
mixed_df = mixed_df.iloc[:, sel.get_support(indices=True)]  # возвращаемое значение будет массивом целых чисел
mixed_df.to_csv('NEW-15-04-merged_flows_p2p_http.csv', sep=',')
keys = mixed_df.keys()

# извлечение тестовых и тренеровочных X и y
X_train, X_test, y_train, y_test = train_test_split(mixed_df.iloc[:, :-1], mixed_df.iloc[:, -1], train_size=0.85)
importances = defaultdict(list)
results_metrics = {}
models = {
    'Дерево решений': DecisionTreeClassifier(random_state=100, max_depth=10),
    'Случайный лес': RandomForestClassifier(n_estimators=31, max_depth=10, random_state=100),
    'Наивный байесовский классификатор': GaussianNB(var_smoothing=1e-7),
    'Логистическая регрессия': LogisticRegression(solver='liblinear', max_iter=700)
}
# Обучение моделей
for key, model in models.items():
    model.fit(X_train, y_train)
    models[key] = model
    results_metrics[key] = eval_model(models[key], X_test, y_test, key)
    if key == 'Дерево решений':
        importances['Дерево решений'] = model.feature_importances_
        fig = plt.figure(figsize=(25, 20))
        _ = tree.plot_tree(model, feature_names=list(X_train.columns.values), class_names=['0', '1'], filled=True)
        fig.savefig("decision_treeNone.png")
    # сохранение признаков и соответствующие значения важности
with open('importances_2.txt', 'w') as f:
    f.writelines([str(keys[np.argsort(importances['Дерево решений'])]), str(importances['Дерево решений'])])


