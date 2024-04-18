from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score




class SVM:  
    def __init__(self):  
        pass

    def generatePredictions(self, X_train, y_train, X_test ):
        # Initialize SVM model
        svm_model = SVC(kernel='linear')

        # Train the SVM model
        svm_model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = svm_model.predict(X_test)

        return y_pred

def evaluate_model(model, y_pred, y_test)
    # Calculate F1 score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, f1