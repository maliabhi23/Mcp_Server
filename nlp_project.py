import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# --- 1. MINI TEXT CLASSIFICATION TASK (from scratch BERT) ---
# Simple dataset
texts = ["I love AI", "I hate bugs", "AI is the future", "I enjoy coding"]
labels = [1, 0, 1, 1]  # 1 = positive, 0 = negative

# Tokenization (very simple)
word_to_idx = {"I": 1, "love": 2, "hate": 3, "AI": 4, "bugs": 5, "is": 6, "the": 7, "future": 8, "enjoy": 9, "coding": 10}
vocab_size = len(word_to_idx) + 1

def tokenize(text):
    return [word_to_idx.get(word, 0) for word in text.split()]

X = [torch.tensor(tokenize(text)) for text in texts]
X_padded = nn.utils.rnn.pad_sequence(X, batch_first=True)
y = torch.tensor(labels)

# Model
class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=10):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        x = self.embed(x).mean(dim=1)
        return self.fc(x)

model = SimpleClassifier(vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(20):
    logits = model(X_padded)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("\nPredictions:", torch.argmax(logits, dim=1).tolist())

# --- 2. SIMPLE POS TAGGER FOR HINDI ---
def simple_hindi_pos_tagger(sentence):
    tags = {"मैं": "PRON", "पढ़ता": "VERB", "हूँ": "AUX", "पानी": "NOUN", "पीता": "VERB"}
    return [(word, tags.get(word, "UNK")) for word in sentence.split()]

print("\nHindi POS Tagging:")
print(simple_hindi_pos_tagger("मैं पढ़ता हूँ"))

# --- 3. ZERNIKE MOMENTS (BASIC VARIANT FROM SCRATCH) ---
def basic_zernike_moment(image):
    image = cv2.resize(image, (64, 64))
    center = (32, 32)
    moments = []
    for n in range(1, 8):  # Simulated moment values
        moment = np.sum((np.indices(image.shape)[0] - center[0])**n + (np.indices(image.shape)[1] - center[1])**n)
        moments.append(moment)
    return moments

img = np.zeros((100, 100), dtype=np.uint8)
cv2.circle(img, (50, 50), 20, 255, -1)
z_moments = basic_zernike_moment(img)
print("\nZernike Moments (simulated):", z_moments)

# --- 4. DALL·E IMAGE GENERATION USING API + REQUESTS ---
def generate_dalle_image(prompt):
    headers = {
        "Authorization": f"Bearer YOUR_OPENAI_API_KEY",
        "Content-Type": "application/json"
    }
    json_data = {
        "prompt": prompt,
        "n": 1,
        "size": "256x256"
    }
    response = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=json_data)
    image_url = response.json()["data"][0]["url"]
    img_data = requests.get(image_url).content
    image = Image.open(BytesIO(img_data))
    image.show()

# Run image generation (optional)
# generate_dalle_image("A robot reading a book")

