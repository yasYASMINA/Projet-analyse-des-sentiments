from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model and necessary objects
nb_model = joblib.load('nb_model.pkl')
lr_model = joblib.load('lr_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')


# Load the accuracy value
with open('accuracy of NB.txt', 'r') as file:
    accuracy = file.read()
with open('accuracy of LR.txt', 'r') as file:
    accuracy = file.read() 

@app.route('/')
def home():
    return render_template('index.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text_tfidf = vectorizer.transform([text])
    prediction = nb_model.predict(text_tfidf)
    prediction = lr_model.predict(text_tfidf)
    predicted_emotion = label_encoder.inverse_transform(prediction)[0]
    return render_template('index.html', prediction_text=f'{predicted_emotion}', accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
