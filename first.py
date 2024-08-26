from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#X is a list of lists
#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male']

#splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)

#decision tree
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(X_train, Y_train)
prediction_tree = clf_tree.predict(X_test)
print(prediction_tree)
accuracy_tree = accuracy_score(Y_test, prediction_tree)

#k nearest neighbors
clf_knn = KNeighborsClassifier()
clf_knn = clf_knn.fit(X_train, Y_train)
prediction_knn = clf_knn.predict(X_test)
print(prediction_knn)
accuracy_knn = accuracy_score(Y_test, prediction_knn)

# Support Vector Classifier
clf_svc = SVC()
clf_svc = clf_svc.fit(X_train, Y_train)
prediction_svc = clf_svc.predict(X_test)
print(prediction_svc)
accuracy_svc = accuracy_score(Y_test, prediction_svc)

accuracies = {
    'Decision Tree': accuracy_tree,
    'K-Nearest Neighbor': accuracy_knn,
    'Support Vector': accuracy_svc
}

best_model = max (accuracies, key=accuracies.get)
print(best_model)