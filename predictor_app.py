from flask import request
from flask import Flask
import flask
from predictor_api import make_prediction, feature_names

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    x_input, predictions = make_prediction(request.args)
    return flask.render_template('predictor.html',
                             x_input=x_input,
                             feature_names=feature_names,
                             prediction=predictions)


if __name__ == '__main__':
    app.run(debug=False)