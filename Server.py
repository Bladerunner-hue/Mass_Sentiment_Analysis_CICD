from flask import Flask, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import main

app = Flask(__name__)

@app.route("/",methods=['GET'])
def code():
    out = open(r'main.py', 'r').read()
    return exec(out)


if __name__ == '__main__':
  app.run(debug=True)