
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class RNN:  
    def __init__(self): 

        pass

    def configModel(input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),

            # Conv1D layer for pattern recognition model and extract the feature from the vectors
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True),

            # GlobalMaxPooling layer to extract relevant features
            tf.keras.layers.GlobalMaxPool1D(),

            # Dense layer with 64 neurons and ReLU activation
            tf.keras.layers.Dense(64, activation='relu'),

            # Dropout layer to prevent overfitting
            tf.keras.layers.Dropout(0.5),

            # Final Dense layer with 1 neuron and sigmoid activation for binary classification
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), metrics=["accuracy"], loss="binary_crossentropy")

        return model

    
    def predict(self,X_test, rnn_model): 

        y_pred_probs = rnn_model.predict(X_test)

        # Threshold probabilities to obtain predicted classes
        y_pred = (y_pred_probs > 0.5).astype(int)
        return y_pred



    def evaluateRNN(self,X_test, y_test, y_pred, rnn_model): 
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_f1_score = f1_score(y_test, y_pred)

        # Evaluate test loss and accuracy
        test_loss, test_accuracy = rnn_model.evaluate(X_test, y_test)
        return test_precision, test_recall, test_f1_score, test_loss, test_accuracy
