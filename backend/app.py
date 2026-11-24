from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend():

    data = request.json

    books = data["books"]

    recommendations = [
        {
            "title": "Dune",
            "score": 0.96
        },
        {
            "title": "Foundation",
            "score": 0.94
        }
    ]

    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)