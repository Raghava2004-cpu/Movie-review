from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('review_prediction.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('a.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text from the request
    data = request.get_json()
    review_text = data['text']
    
    # Make a prediction
    prediction = model.predict([review_text])[0]
    
    # Convert prediction to human-readable format
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    
    # Return the result as a JSON response
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
