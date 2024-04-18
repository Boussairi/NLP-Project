from arabert import ArabertPreprocessor
from transformers import AutoTokenizer, AutoModel
import torch


class ArabertLM: 
    def __init__(self): 
        model_arabert = "aubmindlab/bert-base-arabertv02"
        self.tokenizer = AutoTokenizer.from_pretrained(model_arabert)
        self.model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv02")

    def generate_embeddings(self,text):
        # Tokenize and preprocess the text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # Pass the tokenized input through the model
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Extract the embeddings from the model's output
        embeddings = outputs.last_hidden_state
        return embeddings