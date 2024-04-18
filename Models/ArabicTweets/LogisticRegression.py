from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

class LogisticRegression:
    def __init__(self):
        pass

    def generatePrediction(self, X_train, y_train, X_test): 
        # Create a logistic regression model
        logistic_model = LogisticRegression()

        # Train the model
        logistic_model.fit(X_train, y_train)

        # Predict the labels for the test set
        y_pred = logistic_model.predict(X_test)

        return y_pred
    
    def evaluateModel(self, y_pred, y_test): 
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("F1 Score:", f1)


