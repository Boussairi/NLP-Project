import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = 100
        # Définition de la fonction de perte et de l'optimiseur
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)



    def train(self):
        train_losses = []
        val_losses = []
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            train_losses.append(running_loss / len(self.train_loader))

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                val_losses.append(val_loss / len(self.val_loader))

            if (epoch+1)%20==0:
               print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

        return train_losses, val_losses

def load_data(trainset, valset, testset, batch_size=32):
    train_inputs_tensor = torch.tensor(trainset["tweet_embedding"].tolist(), dtype=torch.float32)
    train_labels_tensor = torch.tensor(trainset["label"].tolist(), dtype=torch.long)
    val_inputs_tensor = torch.tensor(valset["tweet_embedding"].tolist(), dtype=torch.float32)
    val_labels_tensor = torch.tensor(valset["label"].tolist(), dtype=torch.long)
    test_inputs_tensor = torch.tensor(testset["tweet_embedding"].tolist(), dtype=torch.float32)
    test_labels_tensor = torch.tensor(testset["label"].tolist(), dtype=torch.long)
    
    print(len(train_inputs_tensor[0]), len(val_inputs_tensor[0]))
    # Ajustement de la taille de zeros
    if len(train_inputs_tensor[0])> len(val_inputs_tensor[0]):
      zeros = torch.zeros(len(train_inputs_tensor[0]) - len(val_inputs_tensor[0]))
      zeros2 = torch.zeros(len(train_inputs_tensor[0]) - len(test_inputs_tensor[0]))
      val_inputs_tensor = torch.cat([val_inputs_tensor, zeros.unsqueeze(0).expand(val_inputs_tensor.size(0), -1)], dim=1)
      test_inputs_tensor = torch.cat([test_inputs_tensor, zeros2.unsqueeze(0).expand(test_inputs_tensor.size(0), -1)], dim=1)
    
    elif len(train_inputs_tensor[0]) < len(val_inputs_tensor[0]): 
      zeros = torch.zeros(len(val_inputs_tensor[0]) - len(train_inputs_tensor[0]))
      zeros2 = torch.zeros(len(val_inputs_tensor[0]) - len(test_inputs_tensor[0]))
      train_inputs_tensor = torch.cat([train_inputs_tensor, zeros.unsqueeze(0).expand(train_inputs_tensor.size(0), -1)], dim=1)
      test_inputs_tensor = torch.cat([test_inputs_tensor, zeros2.unsqueeze(0).expand(test_inputs_tensor.size(0), -1)], dim=1)

        
    train_dataset = TensorDataset(train_inputs_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_inputs_tensor, val_labels_tensor)
    test_dataset = TensorDataset(test_inputs_tensor, test_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def plot_losses(train_losses, val_losses):
    """
    Affiche la courbe de perte.

    Args:
    - train_losses (list): Liste des pertes d'entraînement pour chaque époque.
    - val_losses (list): Liste des pertes de validation pour chaque époque.
    """
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def evaluate_model(model, test_loader):
    model.eval()
    test_predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    # Calcul des métriques
    precision = precision_score(true_labels, test_predictions, average='macro')
    recall = recall_score(true_labels, test_predictions, average='macro')
    f1 = f1_score(true_labels, test_predictions, average='macro')

    return precision, recall, f1
