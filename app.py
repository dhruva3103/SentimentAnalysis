from flask import abort, Flask, jsonify, request
from flair.models import TextClassifier
from flair.data import Sentence

app = Flask(__name__)

classifier = TextClassifier.load('en-sentiment')


@app.route('/analyzeSentiment', methods=['POST'])
def analyzeSentiment():
    if not request.json or not 'message' in request.json:
        abort(400)
    message = request.json['message']
    sentence = Sentence(message)
    classifier.predict(sentence)
    print('Sentence sentiment: ', sentence.labels)
    label = sentence.labels[0]
    if label.value == 'POSITIVE':
        label.value = 'happy'
    else:
        label.value = 'sad'
    response = {'result': label.value, 'polarity': label.score}
    return jsonify(response), 200


if __name__ == "__main__":
    app.run()