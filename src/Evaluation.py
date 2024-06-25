scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.pipeline import Pipeline 
models = {'Logistic Regression': LogisticRegression(),
         'Decision Tree Classifier': DecisionTreeClassifier(),
         'Random Forest Classifier': RandomForestClassifier(),
         'Gradient Boosting Classifier': GradientBoostingClassifier(),
         #'KNeighbors Classifier': KNeighborsClassifier(n_neighbors =5),
         'SVM': SVC(),
         'SCGD Classifier': SGDClassifier(),
         'Naive_bayes': GaussianNB()}

results = {}

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        classification = classification_report(y_test, pred)
        results[name] = {
            'accuracy': accuracy,
            'classification_report': classification
        }
        print(f"{name} - Accuracy: {accuracy:.4f}")
        print(classification)
    except Exception as e:
        print(f"Error with model {name}: {e}")
