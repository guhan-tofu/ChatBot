from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_message

app = Flask(__name__)
CORS(app)


#@app.route("/", methods=["GET"]) # route for the HTTP GET METHOD for root URL
#def get_index():
#    return render_template("base.html") # returns base html template when accessing the root URL

@app.route("/predict", methods=["POST"]) # route for the HTTP POST METHOD for /predict URL
def predict():
    text = request.get_json().get("message")
    response = get_message(text)
    message = {"answer":response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)


