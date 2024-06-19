import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

import joblib

# Function to load and preprocess data
def load_data(filepath):
    data = pd.read_csv(filepath, encoding='utf-8', delimiter=';', header=None)
    data.columns = ['text', 'emotion']
    return data

# Load the datasets
train_data = load_data("emotions-dataset-for-nlp/train.csv")
test_data = load_data("emotions-dataset-for-nlp/test.csv")
val_data = load_data("emotions-dataset-for-nlp/val.csv")

# Combine train and validation data for training
combined_train_data = pd.concat([train_data, val_data])

# Encode the labels
le = LabelEncoder()
combined_train_data['emotion'] = le.fit_transform(combined_train_data['emotion'])
test_data['emotion'] = le.transform(test_data['emotion'])

# Vectorize the textual data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(combined_train_data['text'])
X_test_tfidf = vectorizer.transform(test_data['text'])

# Train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, combined_train_data['emotion'])

# Entraîner le modèle de régression logistique
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf,combined_train_data['emotion'] )

# Prédictions des deux modèles
y_pred_nb = nb_model.predict(X_test_tfidf)
y_pred_lr = lr_model.predict(X_test_tfidf)


#  Calcul des métriques pour le modèle Naive Bayes
accuracy = nb_model.score(X_test_tfidf, test_data['emotion'])
nb_precision = precision_score(test_data['emotion'], y_pred_nb, average='weighted')
nb_recall = recall_score(test_data['emotion'], y_pred_nb, average='weighted')
nb_f1 = f1_score(test_data['emotion'], y_pred_nb, average='weighted')
nb_confusion = confusion_matrix(test_data['emotion'], y_pred_nb)

# Calcul des métriques pour le modèle de régression logistique

accuracy = lr_model.score(X_test_tfidf, test_data['emotion'])
lr_precision = precision_score(test_data['emotion'], y_pred_lr, average='weighted')
lr_recall = recall_score(test_data['emotion'], y_pred_lr, average='weighted')
lr_f1 = f1_score(test_data['emotion'], y_pred_lr, average='weighted')
lr_confusion = confusion_matrix(test_data['emotion'], y_pred_lr)

# Affichage des résultats
print(f'Accuracy of naive bayes : {accuracy:.2f}')
print(f'Naive Bayes Precision: {nb_precision:.2f}')
print(f'Naive Bayes Recall: {nb_recall:.2f}')
print(f'Naive Bayes F1-Score: {nb_f1:.2f}')
print('Naive Bayes Confusion Matrix:')
print(nb_confusion)

print(f'Accuracy of logistique : {accuracy:.2f}')
print(f'Logistic Regression Precision: {lr_precision:.2f}')
print(f'Logistic Regression Recall: {lr_recall:.2f}')
print(f'Logistic Regression F1-Score: {lr_f1:.2f}')
print('Logistic Regression Confusion Matrix:')
print(lr_confusion)


# Save the accuracy to a file
with open('accuracy of NB.txt', 'w') as file:
    file.write(f'{accuracy:.2f}')
with open('accuracy of LR.txt', 'w') as file:
    file.write(f'{accuracy:.2f}')


# Save the model and necessary objects
joblib.dump(nb_model, 'nb_model.pkl')
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')


print("Model training complete and objects saved.")
