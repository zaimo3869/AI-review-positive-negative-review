# Importer les bibliothèques nécessaires
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
import tkinter as tk
from tkinter import messagebox

# Télécharger les stopwords une fois
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Fonction pour nettoyer le texte en supprimant les caractères spéciaux, 
    les multiples espaces, les majuscules, et les mots vides (stopwords).
    """
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Vérifier si le fichier CSV existe
file_path = 'IMDB Dataset.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please ensure the file is in the correct directory.")

# Charger les données à partir d'un fichier CSV
data = pd.read_csv(file_path)

# Afficher les premières lignes du dataset pour vérifier le chargement des données
print(data.head())

# Appliquer le nettoyage à la colonne de texte
data['cleaned_text'] = data['review'].apply(clean_text)

# Vérifier la distribution des sentiments
print(data['sentiment'].value_counts())

# Initialiser le vecteur TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

# Ajuster et transformer le texte nettoyé en vecteurs TF-IDF
X = vectorizer.fit_transform(data['cleaned_text']).toarray()

# La variable cible (sentiment positif ou négatif)
y = data['sentiment']

# Diviser les données en ensembles d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle de régression logistique
model = LogisticRegression(max_iter=200)

# Entraîner le modèle sur les données d'entraînement
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer l'accuracy du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Afficher un rapport de classification détaillé
print(classification_report(y_test, y_pred))

# Fonction pour faire des prédictions sur de nouvelles critiques
def predict_sentiment(review):
    cleaned_review = clean_text(review)
    vectorized_review = vectorizer.transform([cleaned_review]).toarray()
    prediction = model.predict(vectorized_review)
    return prediction[0]

# Interface graphique avec tkinter
def on_predict():
    review = review_entry.get("1.0", tk.END).strip()
    if review:
        sentiment = predict_sentiment(review)
        result_label.config(text=f"Sentiment: {sentiment}")
    else:
        messagebox.showwarning("Input Error", "Please enter a review.")

# Créer la fenêtre principale
root = tk.Tk()
root.title("IMBD Dataset")

# Créer un champ de saisie pour les critiques
review_label = tk.Label(root, text="Enter your movie review:")
review_label.pack()
review_entry = tk.Text(root, height=10, width=50)
review_entry.pack()

# Créer un bouton pour soumettre la critique
predict_button = tk.Button(root, text="Predict Sentiment", command=on_predict)
predict_button.pack()

# Créer une étiquette pour afficher le résultat
result_label = tk.Label(root, text="Sentiment: ")
result_label.pack()

# Lancer la boucle principale de tkinter
root.mainloop()

