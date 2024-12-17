import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model import BiLSTMModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from collections import Counter
import torch.nn.functional as F

# Définir le dataset personnalisé
class LyricsDataset(Dataset):
    def __init__(self, lyrics, labels, vocab, tokenizer):
        self.lyrics = lyrics
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.lyrics)
    
    def __getitem__(self, idx):
        lyrics = self.lyrics[idx]
        label = self.labels[idx]
        tokenized_lyrics = [self.vocab[token] for token in self.tokenizer(lyrics)]
        return torch.tensor(tokenized_lyrics), torch.tensor(label)

# Charger les données
df = pd.read_csv("paroles_artistes.csv")  # Charger le fichier CSV téléchargé
# Encoder les labels (Artiste : Poopy ou Bodo)
df['label'] = LabelEncoder().fit_transform(df['label'])  # Convertir les labels en numérique (0 ou 1)

# Tokenisation et création du vocabulaire
tokenizer = get_tokenizer("basic_english")
counter = Counter()
for lyrics in df['lyrics']:
    counter.update(tokenizer(lyrics))
vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.items())}  # Ajouter un index pour chaque mot
vocab['<pad>'] = 0  # Index pour padding
vocab['<unk>'] = 1  # Index pour unknwon words

# Séparer les données en train et test
X_train, X_test, y_train, y_test = train_test_split(df['lyrics'], df['label'], test_size=0.2, random_state=42)

# Créer des DataLoaders
train_dataset = LyricsDataset(X_train.values, y_train.values, vocab, tokenizer)
test_dataset = LyricsDataset(X_test.values, y_test.values, vocab, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialiser le modèle
input_dim = len(vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 2  # "poopy" ou "bodo"
n_layers = 2
dropout = 0.5

model = BiLSTMModel(input_dim, embedding_dim, hidden_dim, output_dim, n_layers, dropout)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Fonction pour entraîner le modèle
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# Fonction pour évaluer le modèle
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f'Accuracy on test data: {accuracy*100:.2f}%')

# Entraîner le modèle
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Évaluer le modèle
evaluate_model(model, test_loader)


