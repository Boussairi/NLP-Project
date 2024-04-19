import torch
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

class HardVotingClassifier:
    def __init__(self, svm_model, lstm_model, lstm, cnn_model):
        self.svm_model = svm_model
        self.lstm_model = lstm_model
        self.lstm = lstm
        self.cnn_model = cnn_model

    def predict(self, X_val, X_val_lstm, val_inputs_tensor):
        svm_pred = self.svm_model.predict_svm(X_val)
        lstm_pred = self.lstm_model.lstm_predict(self.lstm, X_val_lstm)
        lstm_pred_binary = (lstm_pred > 0.5).astype(int)
        lstm_prediction = lstm_pred_binary.ravel()
        with torch.no_grad():
            cnn_pred = self.cnn_model(val_inputs_tensor)
            cnn_pred = torch.argmax(torch.softmax(cnn_pred, dim=1), dim=1).numpy()

        voting_pred = []
        for i in range(len(X_val)):
            pred = np.argmax(np.bincount([svm_pred[i], lstm_prediction[i], cnn_pred[i]]))
            voting_pred.append(pred)

        return voting_pred

def prepare_data(trainset, valset):
    # données pour CNN
    X_train_tensor = torch.tensor(trainset["tweet_embedding"].tolist(), dtype=torch.float32)
    val_inputs_tensor = torch.tensor(valset["tweet_embedding"].tolist(), dtype=torch.float32)
    # Ajustement de la taille de zeros
    if len(X_train_tensor[0])> len(val_inputs_tensor[0]):
      zeros = torch.zeros(len(X_train_tensor[0]) - len(val_inputs_tensor[0]))
      val_inputs_tensor = torch.cat([val_inputs_tensor, zeros.unsqueeze(0).expand(val_inputs_tensor.size(0), -1)], dim=1)    
    elif len(X_train_tensor[0]) < len(val_inputs_tensor[0]): 
      zeros = torch.zeros(len(val_inputs_tensor[0]) - len(X_train_tensor[0]))
      X_train_tensor = torch.cat([X_train_tensor, zeros.unsqueeze(0).expand(X_train_tensor.size(0), -1)], dim=1)


   
    # données pour LSTM
    X_train_lstm = tf.constant(trainset["tweet_embedding"].tolist(), dtype=tf.float32)
    X_val_lstm = tf.constant(valset["tweet_embedding"].tolist(), dtype=tf.float32)

    #données pour SVM
    X_train = tf.constant(trainset["tweet_embedding"].tolist(), dtype=tf.float32)
    X_val = tf.constant(valset["tweet_embedding"].tolist(), dtype=tf.float32)
    y_val = tf.constant(valset["label"].tolist(), dtype=tf.int32)
    max_dim = max(X_train.shape[1], X_val.shape[1])
    X_train = np.pad(X_train, ((0, 0), (0, max_dim - X_train.shape[1])), mode='constant')
    X_val = np.pad(X_val, ((0, 0), (0, max_dim - X_val.shape[1])), mode='constant')




    return val_inputs_tensor, X_val, y_val, X_val_lstm

