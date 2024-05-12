from flask import render_template, request, Blueprint
from nb_classifier import NaiveBayesClassifier

# Load the spam classifier
naive_bayes_classifier, train_metrics = NaiveBayesClassifier.load_classifier("../models/naive_bayes_classifier.pkl",
                                                                             "../data/spam_NLP.csv")
# Create a blueprint for routes
bp = Blueprint("routes", __name__)


@bp.route('/')
def index():
    """
    Render the index page.

    Returns:
        str: Rendered HTML content for the index page.
    """
    return render_template('index.html')


@bp.route("/classify", methods=["POST"])
def classify_email():
    """
    Classify an email as spam or not spam using the selected classifier.

    Returns:
        str: Rendered HTML content for the result page.
    """
    if request.method == "POST":
        email_text = request.form["email_text"]
        selected_classifier = request.form["classifier"]

        if selected_classifier == "naive_bayes":
            prediction = naive_bayes_classifier.predict(email_text)
        elif selected_classifier == "not_implemented_yet":
            return render_template("404.html")
        else:
            return render_template("404.html")

        if prediction == 1:
            result = "Spam"
        else:
            result = "Not Spam"
        return render_template("result.html", result=result, metrics=train_metrics)
