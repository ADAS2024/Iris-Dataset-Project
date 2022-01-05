import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('iris.csv')
data = data.drop('Id', axis=1)


def main():
    datavisual()
    Irisprogram()


def Irisprogram():
    X = data.drop(['Species'], axis=1)
    y = data['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
    k_range = list(range(1, 26))
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores.append(metrics.accuracy_score(y_test, y_pred))

    knn = KNeighborsClassifier(n_neighbors=12)
    knn.fit(X.values, y)
    slength = int(input('What is your Iris Sepal Length?'))
    swidth = int(input('What is your Iris Sepal Width?'))
    plength = int(input('What is your Iris Petal Length?'))
    pwidth = int(input('What is your Iris Sepal Width?'))
    print()
    print("Predicted Flower:" + " " + knn.predict([[slength, swidth, plength, pwidth]]))


def datavisual():
    sns.pairplot(data, hue='Species')
    plt.show()
    print()
    sns.violinplot(y='Species', x='SepalLengthCm', data=data)
    plt.show()
    sns.violinplot(y='Species', x='SepalWidthCm', data=data)
    plt.show()
    sns.violinplot(y='Species', x='PetalLengthCm', data=data)
    plt.show()
    sns.violinplot(y='Species', x='PetalWidthCm', data=data)
    plt.show()


# Press the green button in the gutter to run the script.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

main()
