# coding=utf-8

from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def draw():
    plt.figure(figsize=(35, 10))
    tree.plot_tree(
        clf, fontsize=9, feature_names=wine.feature_names, class_names=wine.target_names
    )
    plt.show()


wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.3
)

clf = tree.DecisionTreeClassifier(
    criterion="entropy",
    random_state=0,
    splitter="best",
    ccp_alpha=0,
    # max_depth=3,
    # min_samples_leaf=10,
    # min_samples_split=10,
)
clf = clf.fit(X_train, y_train)
print("score:", clf.score(X_test, y_test))


# draw()


pruning_path = clf.cost_complexity_pruning_path(X_train, y_train)
clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=0.1)
clf = clf.fit(X_train, y_train)
print("score:", clf.score(X_test, y_test))
