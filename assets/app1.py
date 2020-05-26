from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__,
            template_folder="templates")

@app.route('/')
def index():
  return render_template("dataset.html")


if __name__ == '__main__':
    app.run(debug=True)