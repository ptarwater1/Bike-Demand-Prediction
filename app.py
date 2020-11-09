# import the Flask class from the flask module
from flask import Flask, render_template, redirect, url_for, request, session, flash
from functools import wraps
import numpy as np
import pickle
import math

# create the application object
app = Flask(__name__)

# config
app.secret_key = 'bike0123'

# login required decorator
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('You need to login first.')
            return redirect(url_for('login'))
    return wrap


@app.route('/')
@login_required
def home():
    return render_template('predictor.html')


@app.route('/data-visualization')
def datavisualization():
    return render_template('data-visualization.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            session['logged_in'] = True
            flash('You were logged in.')
            return redirect(url_for('home'))
    return render_template('login.html', error=error)


@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    flash('You were logged out.')
    return redirect(url_for('login'))


pred_model = pickle.load(open('pred_model.pkl', 'rb'))


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    if int_features[3] == 1:
        int_features.append(0)
        int_features.append(0)
        int_features.append(0)
    if int_features[3] == 2:
        del int_features[3]
        int_features.append(0)
        int_features.append(1)
        int_features.append(0)
        int_features.append(0)
    if int_features[3] == 3:
        del int_features[3]
        int_features.append(0)
        int_features.append(0)
        int_features.append(1)
        int_features.append(0)
    if int_features[3] == 4:
        del int_features[3]
        int_features.append(0)
        int_features.append(0)
        int_features.append(0)
        int_features.append(1)
    final = [np.array(int_features, dtype=float)]
    prediction = pred_model.predict(final)
    output = math.ceil(prediction[0])

    print(final)
    print
    return render_template('predict.html',
                           pred='The predicted amount of bikes for any given hour: {} Bikes'.format(output))




if __name__ == '__main__':
    app.run(debug=True)