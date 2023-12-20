from flask import Flask, render_template, request, jsonify
from chat import get_response

app = Flask(__name__)

@app.get("/")
def index_get():
    """
    Endpoint: GET '/'
    
    Renders the home page for user interaction.
    """
    return render_template("base.html")

@app.post("/predict")
def predict():
    """
    Endpoint: POST '/predict'
    
    Handles POST requests for user input and returns the chatbot's response.
    
    Payload:
        - JSON with a 'message' field containing the user input.
    
    Returns:
        - JSON response containing the chatbot's answer.
    """
    text = request.get_json().get("message")
    
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)