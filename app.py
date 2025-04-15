from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the tokenizer and model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model('word_predictor.keras')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']

    # Tokenize and pad the input text
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded_token_text = pad_sequences([token_text], maxlen=1401, padding='pre')

    # Predict the next word
    word_num = np.argmax(model.predict(padded_token_text), axis=-1)[0]

    # Map the predicted number back to the word
    predicted_word = None
    for word, index in tokenizer.word_index.items():
        if index == word_num:
            predicted_word = word
            break

    return jsonify({'suggestion': predicted_word})

if __name__ == '__main__':
    app.run(debug=True)