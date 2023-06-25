# NAME @: Priyanka Sawant
# TOPIC @: Online Payment Fraud Detection Model Deployment
# DATE @:  11/06/2023


# IMPORT THE DEPENDENCIES

import pickle
import joblib
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm
import numpy as np
from wtforms import (StringField, BooleanField, DateTimeField,
                     RadioField, SelectField, TextField,
                     TextAreaField, SubmitField)

# REMEMBER TO LOAD THE MODEL AND THE SCALER!

# 1. LOAD THE SVC MODEL
pkl_filename = "Models/RF_pickle_model.pkl"
with open(pkl_filename, 'rb') as file2:
    RF_pickle_model = pickle.load(file2)

# 2. LOAD THE SCLAER OBJECT 
RF_scaler = joblib.load("Models/RF_scaler.pkl")


# 3. CREATE A PREICTION FUNCTION
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

    return classes[class_ind[0]]


# 4 . CREATE A FLASK FORM

class InfoForm(FlaskForm):
    """
    This general class gets a lot of form data about user.
    Mainly a way to go through many of the WTForms Fields.
    """

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
