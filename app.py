

# NAME @: Priyanka Sawant
# TOPIC @: Online Payment Fraud Detection Model Deployment
# DATE @:  11/06/2023


# IMPORT THE DEPENDENCIES 

from flask import (Flask, 
                   render_template, 
                   session, redirect, 
                   url_for, request,
                   jsonify)
 
from helper_functions import (RF_pickle_model,
                              RF_scaler,
                              return_prediction,
                              InfoForm)




app = Flask(__name__)
# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'mysecretkey'



# 1. View point to show the form and collect the data from user 
@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = InfoForm()
    
    # If the form is valid on submission 
    if form.validate_on_submit():

        # Grab the data from the  form.
        session['step'] = form.step.data
        session['type'] = form.type.data
        session['amount'] = form.amount.data
        session['sender_OldBal'] = form.sender_OldBal.data
        session['sender_NewBal'] = form.sender_NewBal.data
        session['recipient_OldBal'] = form.recipient_OldBal.data
        session['recipient_NewBal'] = form.recipient_NewBal.data

        # Redirect the data to Predictioin function
        return redirect(url_for("prediction"))

    # Show the form for first visit 
    return render_template('home.html', form=form)


# 2. View point to show the result 
@app.route('/prediction')
def prediction():
    
    content = {}

    content['step'] = float(session['step'])
    content['type'] = float(session['type'])
    content['amount'] = float(session['amount'])
    content['sender_OldBal'] = float(session['sender_OldBal'])
    content['sender_NewBal'] = float(session['sender_NewBal'])
    content['recipient_OldBal'] = float(session['recipient_OldBal'])
    content['recipient_NewBal'] = float(session['recipient_NewBal'])



     # PRINT THE DATA PRESENT IN THE REQUEST 
    print("[INFO] WEB Request  - " , content)


    # Actual prediction done by this function 
    results = return_prediction(model=RF_pickle_model, scaler=RF_scaler, sample_json=content)


    # PRINT THE RESULT 
    print("[INFO] WEB Responce - " , results)

    return render_template('thankyou.html', results=results)

# 3. View point to handle the restfull api for prediciton 
@app.route('/api/prediction', methods=['POST'])
def predict_flower():
    
    # RECIEVE THE REQUEST 
    content = request.json
    
    # PRINT THE DATA PRESENT IN THE REQUEST 
    print("[INFO] API Request - " , content)
    
    # PREDICT THE CLASS USING HELPER FUNCTION 
    results = return_prediction(model=RF_pickle_model,scaler=RF_scaler,sample_json=content)
    
    # PRINT THE RESULT 
    print("[INFO] API Responce - " , results)
          
    # SEND THE RESULT AS JSON OBJECT 
    return jsonify(results)




# 4. View Point To handle the 404 Not found Error 
@app.errorhandler(404)
def page_not_found(e):
    return render_template('notfound.html'), 404


# if __name__ == '__main__':
#     app.run('0.0.0.0',8080,debug=False)


if __name__ == '__main__':
    app.run()
