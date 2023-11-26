import re
from collections import Counter
import networkx as nx
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class AlternativeRAKE:
    def __init__(self, stopwords):
        self.stopwords = set(stopwords)

    def run(self, text):
        sentences = re.split(r'[.!?]', text)
        phrases = self._get_phrases(sentences)
        scores = self._calculate_scores(phrases)

        sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        keywords = [(kw[0], kw[1]) for kw in sorted_keywords if len(kw[0].split(' ')) < 4 and len(kw[0]) < 30]

        return keywords

    def _get_phrases(self, sentences):
        phrases = []

        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            phrase = ' '.join([word for word in words if word not in self.stopwords])
            if phrase:
                phrases.append(phrase)

        return phrases

    def _calculate_scores(self, phrases):
        scores = Counter()

        for phrase in phrases:
            words = phrase.split()
            phrase_length = len(words)
            degree = phrase_length - 1 if phrase_length > 1 else 1

            for word in words:
                scores[word] += degree / phrase_length

        return scores

class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

def train_regression_model(X, y):
    model = Sequential()
    model.add(Dense(1, input_dim=X.shape[1], activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=100, verbose=0)
    return model

def main():
    # Example text
    stopwords = ["and", "the", "is", "in", "it"]
    text = "This is a sample text. The goal is to extract important keywords using the RAKE algorithm."

    # Keyword extraction using AlternativeRAKE
    rake = AlternativeRAKE(stopwords)
    keywords = rake.run(text)
    print("Keywords:", keywords)

    # Regression example with PyTorch
    X = torch.randn(100, 2)
    y = 3 * X[:, 0] + 2 * X[:, 1] + torch.randn(100)
    regression_model = RegressionModel(input_size=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(regression_model.parameters(), lr=0.01)

    for epoch in range(100):
        y_pred = regression_model(X)
        loss = criterion(y_pred, y.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("PyTorch Regression Model Weights:", regression_model.state_dict())

    # Regression example with Keras
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.numpy())
    model = train_regression_model(X_scaled, y.numpy())
    keras_weights = model.get_weights()
    print("Keras Regression Model Weights:", keras_weights)

    # Graph example with NetworkX
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    print("NetworkX Graph Nodes:", G.nodes())
    print("NetworkX Graph Edges:", G.edges())

if __name__ == "__main__":
    main()
