from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
import pickle
import joblib
import numpy as np
from wtforms import (StringField, BooleanField, DateTimeField,
                     RadioField, SelectField, TextField,
                     TextAreaField, SubmitField)
from wtforms.validators import DataRequired

app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'mysecretkey'


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class InfoForm(FlaskForm):
    """This general class to accept form data"""

    step = StringField('Enter Your Step', validators=[DataRequired()])

    type = RadioField('Please choose your type of payment done:',
                      choices=[('5', 'CASH_IN'), ('4', 'Transfer'), ('3', 'Payment'), ('2', 'DEBIT'),
                               ('1', 'CASH_OUT')])

    amount = StringField('Enter Amount', validators=[DataRequired()])

    sender_OldBal = StringField('Enter Sender Old Balance', validators=[DataRequired()])

    sender_NewBal = StringField('Enter Sender New Balance', validators=[DataRequired()])

    recipient_OldBal = StringField('Enter Recipient Old Balance', validators=[DataRequired()])

    recipient_NewBal = StringField('Enter Recipient New Balance', validators=[DataRequired()])

    submit = SubmitField('Submit')


def return_prediction(model, scaler, sample_json):
    # For larger data features, you should probably write a for loop
    # That builds out this array for you

    step = sample_json['step']
    typ = sample_json['type']
    amt = sample_json['amount']
    sob = sample_json['sender_OldBal']
    snb = sample_json['sender_NewBal']
    rob = sample_json['recipient_OldBal']
    rnb = sample_json['recipient_NewBal']

    person = [[step, typ, amt, sob, snb, rob, rnb]]

    person = scaler.transform(person)

    classes = np.array(['Not-Fraud:- 0', 'Fraud:- 1'])

    class_ind = model.predict(person)

    return classes[class_ind]


# REMEMBER TO LOAD THE MODEL AND THE SCALER!

# LOAD THE SVC MODEL
pkl_filename = "Models/RF_pickle_model.pkl"
with open(pkl_filename, 'rb') as file2:
    RF_pickle_model = pickle.load(file2)

# LOAD THE SCLAER OBJECT 
RF_scaler = joblib.load("Models/RF_scaler.pkl")


@app.route('/', methods=['GET', 'POST'])
def index():
    # Create instance of the form.
    form = InfoForm()
    # If the form is valid on submission 
    if form.validate_on_submit():
        # Grab the data from  form.

        session['step'] = form.step.data
        session['type'] = form.type.data
        session['amount'] = form.amount.data
        session['sender_OldBal'] = form.sender_OldBal.data
        session['sender_NewBal'] = form.sender_NewBal.data
        session['recipient_OldBal'] = form.recipient_OldBal.data
        session['recipient_NewBal'] = form.recipient_NewBal.data

        return redirect(url_for("prediction"))

    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():
    content = {}

    content['step'] = float(session['step'])
    content['type'] = float(session['type'])
    content["amount"] = float(session['amount'])
    content['sender_OldBal'] = float(session['sender_OldBal'])
    content['sender_NewBal'] = float(session['sender_NewBal'])
    content['recipient_OldBal'] = float(session['recipient_OldBal'])
    content['recipient_NewBal'] = float(session['recipient_NewBal'])

    results = return_prediction(model=RF_pickle_model, scaler=RF_scaler, sample_json=content)

    return render_template('thankyou.html', results=results)


# if __name__ == '__main__':
#     app.run('0.0.0.0',port = 8080 , debug=True)


if __name__ == '__main__':
    app.run()
