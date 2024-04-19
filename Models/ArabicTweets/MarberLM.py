from transformers import AutoTokenizer, AutoModel
import torch



class MarbertLM:
    def __init__(self):
        self.MARBERT_tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
        self.MARBERT_model = AutoModel.from_pretrained("UBC-NLP/MARBERT") 

    def generate_embeddings(self,text):
        # Tokenize and preprocess the text
        inputs = self.MARBERT_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # Pass the tokenized input through the model
        with torch.no_grad():
            outputs = self.MARBERT_model(**inputs)
        # Extract the embeddings from the model's output
        embeddings = outputs.last_hidden_state
        return embeddings