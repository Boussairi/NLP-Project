import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score

class LSTMModel:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def f1_metric(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.round(y_pred)
        return tf.py_function(f1_score, (y_true, y_pred), tf.float32)

    def train(self, trainset, testset):
        # Création du modèle LSTM
        lstm = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=len(self.tokenizer.vocab), output_dim=8),
            tf.keras.layers.LSTM(8),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dropout(0.5),  # Ajouter une couche de dropout pour la régularisation
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        # Compilation du modèle LSTM
        lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', self.f1_metric])
        # Affichage du résumé du modèle
        print(lstm.summary())
        # Convertir les données d'entraînement et de test en tenseurs TensorFlow
        X_train = tf.constant(trainset["tweet_embedding"].tolist(), dtype=tf.float32)
        y_train = tf.constant(trainset["label"].tolist(), dtype=tf.int32)
        X_test = tf.constant(testset["tweet_embedding"].tolist(), dtype=tf.float32)
        y_test = tf.constant(testset["label"].tolist(), dtype=tf.int32)
        # Ajustez la taille des séquences
        if len(X_train[0]) > len(X_test[0]):
            pad_width = [(0, 0), (0, len(X_train[0]) - len(X_test[0]))]
            X_test = tf.pad(X_test, pad_width, constant_values=0)
        elif len(X_train[0]) < len(X_test[0]):
            pad_width = [(0, 0), (0, len(X_test[0]) - len(X_train[0]))]
            X_train = tf.pad(X_train, pad_width, constant_values=0)
        # Entraînement du modèle LSTM
        lstm.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), class_weight={1:4, 0:1})

        return lstm
    
    def evaluate_lstm(self, lstm, testset):
        X_test = tf.constant(testset["tweet_embedding"].tolist(), dtype=tf.float32)
        y_test = tf.constant(testset["label"].tolist(), dtype=tf.int32)

        y_pred = lstm.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)

        print("Accuracy:", accuracy)
        print("F1-score:", f1)

    def lstm_predict(self, lstm, X_test):
       y_pred = lstm.predict(X_test)
       return y_pred
    
