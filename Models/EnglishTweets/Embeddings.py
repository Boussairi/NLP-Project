import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch import nn
from transformers import Trainer,TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import AutoModel, DebertaV2Tokenizer




class Embeddings():
   def __init__(self) :
        pass
   
   def bert(self, train_tweets, val_tweets, test_tweets):
      tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', num_labels=2,
                                                loss_function_params={"weight": [0.75, 0.25]})

      train_encodings = tokenizer(train_tweets, padding=True, truncation=True, max_length=512)
      val_encodings = tokenizer(val_tweets, padding=True, truncation=True, max_length=512)
      test_encodings = tokenizer(test_tweets, padding=True, truncation=True, max_length=512)

      return tokenizer, train_encodings, val_encodings, test_encodings

   def create_dataset(self, encodings, labels):
      """
      Crée un dataframe à partir d'un ensemble d'encodages et de libellés.
      """
      dataset = pd.DataFrame({
         "tweet_embedding": encodings,
         "label": labels
      })
      return dataset
      

   def roberta(self, train_tweets, val_tweets, test_tweets):
      task='sentiment'
      MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
      tokenizer = AutoTokenizer.from_pretrained(MODEL, num_labels=2, loss_function_params={"weight": [0.75, 0.25]})

      train_encodings = tokenizer(train_tweets, padding=True, truncation=True, max_length=512)
      val_encodings = tokenizer(val_tweets, padding=True, truncation=True, max_length=512)
      test_encodings = tokenizer(test_tweets, padding=True, truncation=True, max_length=512)

      return tokenizer, train_encodings, val_encodings, test_encodings


   def create_dataframe(self, embeddings, labels):
      """
      Crée un DataFrame à partir d'une liste d'embeddings et de leurs labels correspondants.
      """
      # Créer une liste pour stocker les données
      df = []
      # Itérer sur chaque embedding et son label correspondant
      for emb, label in zip(embeddings, labels):
         # Ajouter une nouvelle ligne avec l'embedding et le label correspondant
         df.append({'tweet_embedding': emb, 'label': label})
      # Créer le DataFrame à partir de la liste de dictionnaires
      df = pd.DataFrame(df)
      return df
   

   def deberta(self, train_tweets, val_tweets, test_tweets):
      
      model = AutoModel.from_pretrained("microsoft/DeBERTa-V3-XSmall")
      tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/DeBERTa-V3-XSmall")

      train_encodings = tokenizer(train_tweets, padding=True, truncation=True, max_length=512)
      val_encodings = tokenizer(val_tweets, padding=True, truncation=True, max_length=512)
      test_encodings = tokenizer(test_tweets, padding=True, truncation=True, max_length=512)

      return tokenizer, train_encodings, val_encodings, test_encodings
   





   
   


