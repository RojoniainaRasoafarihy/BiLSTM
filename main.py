import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from model import BiLSTMClassifier
from tqdm import tqdm
import os


# Charger les données
df = pd.read_csv("paroles_artistes.csv")  # Charger le fichier CSV téléchargé
# Encoder les labels (Artiste : Poopy ou Bodo)
le = LabelEncoder()
df['Artiste'] = le.fit_transform(df['Artiste'])  # Poopy -> 0, Bodo -> 1

# Encoder les labels (Artiste : Poopy ou Bodo)
le = LabelEncoder()
df['Artiste'] = le.fit_transform(df['Artiste'])  # Poopy -> 0, Bodo -> 1

# Encodage des paroles en indices dans un vocabulaire
all_words = [word for text in df['Paroles'] for word in text.split()]
vocab = {word: idx+1 for idx, (word, _) in enumerate(set(all_words))}  # Chaque mot a un indice unique

# Fonction de padding des séquences
MAX_LENGTH = 100  # Longueur maximale de chaque séquence

def pad_sequence(text, vocab, max_length=MAX_LENGTH):
    words = text.split()
    indexed_words = [vocab.get(word, 0) for word in words]  # Encodage des mots
    padded = indexed_words[:max_length]
    return padded + [0] * (max_length - len(padded))

df['Padded'] = df['Paroles'].apply(lambda x: pad_sequence(x, vocab))

# Dataset et DataLoader
class LyricsDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = torch.tensor(self.data['Padded'].iloc[idx], dtype=torch.long)
        label = torch.tensor(self.data['Artiste'].iloc[idx], dtype=torch.long)
        return tokens, label

# Split des données
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = LyricsDataset(train_data)
test_dataset = LyricsDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Initialisation du modèle
VOCAB_SIZE = len(vocab) + 1  # Ajouter 1 pour le padding
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 2  # Poopy ou Bodo
N_LAYERS = 2
DROPOUT = 0.5

model = BiLSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)

# Optimiseur et fonction de perte
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Entraînement du modèle
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in tqdm(iterator):
        text, labels = batch
        optimizer.zero_grad()
        output = model(text)
        
        loss = criterion(output, labels)
        acc = accuracy_score(labels.numpy(), torch.argmax(output, 1).numpy())
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Fonction d'évaluation
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in tqdm(iterator):
            text, labels = batch
            output = model(text)
            
            loss = criterion(output, labels)
            acc = accuracy_score(labels.numpy(), torch.argmax(output, 1).numpy())
            
            epoch_loss += loss.item()
            epoch_acc += acc
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Entraînement et évaluation sur plusieurs époques
NUM_EPOCHS = 5
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, test_loader, criterion)
    
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    print(f'Training Loss: {train_loss:.3f}, Training Accuracy: {train_acc:.3f}')
    print(f'Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_acc:.3f}')

