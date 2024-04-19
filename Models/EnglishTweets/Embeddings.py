import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from torch import nn
from transformers import Trainer,TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import AutoModel, DebertaV2Tokenizer


class Embeddings():
   def __init__(self) :
        pass
  
   def bert(self, train_tweets, val_tweets, test_tweets):
        # Initialize the BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', num_labels=2,
                                                loss_function_params={"weight": [0.75, 0.25]})
        
        # Tokenize the input tweets and pad/truncate them to a maximum length of 512 tokens
        train_encodings = tokenizer(train_tweets, padding=True, truncation=True, max_length=512)
        val_encodings = tokenizer(val_tweets, padding=True, truncation=True, max_length=512)
        test_encodings = tokenizer(test_tweets, padding=True, truncation=True, max_length=512)


        # Convert the encodings to tensors
        input_ids_train = torch.tensor(train_encodings['input_ids'])
        attention_mask_train = torch.tensor(train_encodings['attention_mask'])
        input_ids_val = torch.tensor(val_encodings['input_ids'])
        attention_mask_val = torch.tensor(val_encodings['attention_mask'])
        input_ids_test = torch.tensor(test_encodings['input_ids'])
        attention_mask_test = torch.tensor(test_encodings['attention_mask'])

        # Load the pre-trained BERT model
        model = BertModel.from_pretrained('bert-base-uncased')

        # Pass the tokenized input through the model
        with torch.no_grad():
            outputs_train = model(input_ids=input_ids_train, attention_mask=attention_mask_train)
            outputs_val = model(input_ids=input_ids_val, attention_mask=attention_mask_val)
            outputs_test = model(input_ids=input_ids_test, attention_mask=attention_mask_test)

        # Extract the embeddings from the model's output
        embeddings_train = outputs_train.last_hidden_state
        embeddings_val = outputs_val.last_hidden_state
        embeddings_test = outputs_test.last_hidden_state

        return tokenizer, embeddings_train, embeddings_val, embeddings_test
   

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
        task = 'sentiment'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
        
        # Initialize the RoBERTa tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        
        # Tokenize the input tweets and pad/truncate them to a maximum length of 512 tokens
        train_encodings = tokenizer(train_tweets, padding=True, truncation=True, max_length=512)
        val_encodings = tokenizer(val_tweets, padding=True, truncation=True, max_length=512)
        test_encodings = tokenizer(test_tweets, padding=True, truncation=True, max_length=512)

        # Convert the encodings to tensors
        input_ids_train = torch.tensor(train_encodings['input_ids'])
        attention_mask_train = torch.tensor(train_encodings['attention_mask'])
        input_ids_val = torch.tensor(val_encodings['input_ids'])
        attention_mask_val = torch.tensor(val_encodings['attention_mask'])
        input_ids_test = torch.tensor(test_encodings['input_ids'])
        attention_mask_test = torch.tensor(test_encodings['attention_mask'])

        # Load the pre-trained RoBERTa model
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)

        # Pass the tokenized input through the model
        with torch.no_grad():
            outputs_train = model(input_ids=input_ids_train, attention_mask=attention_mask_train)
            outputs_val = model(input_ids=input_ids_val, attention_mask=attention_mask_val)
            outputs_test = model(input_ids=input_ids_test, attention_mask=attention_mask_test)

        # Extract the embeddings from the model's output
        embeddings_train = outputs_train.last_hidden_state
        embeddings_val = outputs_val.last_hidden_state
        embeddings_test = outputs_test.last_hidden_state

        return tokenizer, embeddings_train, embeddings_val, embeddings_test


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
        # Charger le modèle et le tokenizer DeBERTa
        model = AutoModel.from_pretrained("microsoft/DeBERTa-V3-XSmall")
        tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/DeBERTa-V3-XSmall")

        # Tokenizer les tweets d'entraînement, de validation et de test
        train_encodings = tokenizer(train_tweets, padding=True, truncation=True, max_length=512)
        val_encodings = tokenizer(val_tweets, padding=True, truncation=True, max_length=512)
        test_encodings = tokenizer(test_tweets, padding=True, truncation=True, max_length=512)

        # Convertir les encodages en tensors
        input_ids_train = torch.tensor(train_encodings['input_ids'])
        attention_mask_train = torch.tensor(train_encodings['attention_mask'])
        input_ids_val = torch.tensor(val_encodings['input_ids'])
        attention_mask_val = torch.tensor(val_encodings['attention_mask'])
        input_ids_test = torch.tensor(test_encodings['input_ids'])
        attention_mask_test = torch.tensor(test_encodings['attention_mask'])

        # Passer les entrées tokenisées à travers le modèle
        with torch.no_grad():
            outputs_train = model(input_ids=input_ids_train, attention_mask=attention_mask_train)
            outputs_val = model(input_ids=input_ids_val, attention_mask=attention_mask_val)
            outputs_test = model(input_ids=input_ids_test, attention_mask=attention_mask_test)

        # Extraire les embeddings à partir de la dernière couche cachée du modèle
        embeddings_train = outputs_train.last_hidden_state
        embeddings_val = outputs_val.last_hidden_state
        embeddings_test = outputs_test.last_hidden_state

        return tokenizer, embeddings_train, embeddings_val, embeddings_test 

