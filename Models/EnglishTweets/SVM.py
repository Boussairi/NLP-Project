from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class SVMModel:
    def __init__(self, kernel='linear', max_iter=100):
        self.kernel = kernel
        self.max_iter = max_iter
        self.model = None

    def train(self, X_train, y_train):
        self.model = SVC(kernel=self.kernel, max_iter=self.max_iter)
        self.model.fit(X_train, y_train)
    
    def predict_svm(self, X_val):
      y_pred = self.model.predict(X_val)
      return y_pred


    def evaluate(self, X_val, y_val):
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        return accuracy, f1

def prepare_data_svm(trainset, valset):
    X_train = np.array(trainset["tweet_embedding"].tolist(), dtype=np.float32)
    y_train = np.array(trainset["label"], dtype=np.int64)
    X_val = np.array(valset["tweet_embedding"].tolist(), dtype=np.float32)
    y_val = np.array(valset["label"], dtype=np.int64)
    max_dim = max(X_train.shape[1], X_val.shape[1])
    X_train = np.pad(X_train, ((0, 0), (0, max_dim - X_train.shape[1])), mode='constant')
    X_val = np.pad(X_val, ((0, 0), (0, max_dim - X_val.shape[1])), mode='constant')
    return X_train, y_train, X_val, y_val



