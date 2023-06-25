from flask import Flask, request, jsonify
import numpy as np  
import pickle
import joblib


#### THIS IS WHAT WE DO IN POSTMAN ###
# STEP 1: Create New Request
# STEP 2: Select POST
# STEP 3: Type correct URL (http://127.0.0.1:5000/prediction)
# STEP 4: Select Body
# STEP 5: Select raw and then JSON type
# STEP 6: Type or Paste in example json request
# STEP 7: Run 01-Basic-API.py to launch server and confirm the site is running
# Step 8: Run API request

### IMP NOTES
# Set localhost = '0.0.0.0' and port = 8080 in 01-Basic-API.py 
# To accept the request from other client over a wifi Connection 

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

# LOAD THE RF MODEL
pkl_filename = "Models/RF_pickle_model.pkl"
with open(pkl_filename, 'rb') as file2:
    RF_pickle_model = pickle.load(file2)

# LOAD THE SCLAER OBJECT 
RF_scaler = joblib.load("Models/RF_scaler.pkl")



app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>FLASK APP IS RUNNING!</h1>'


@app.route('/prediction', methods=['POST'])
def predict_flower():
    
    # RECIEVE THE REQUEST 
    content = request.json
    
    # PRINT THE DATA PRESENT IN THE REQUEST 
    print("[INFO] Request: ", content)
    
    # PREDICT THE CLASS USING HELPER FUNCTION 
    results = return_prediction(model=RF_pickle_model,scaler=RF_scaler,sample_json=content)
    
    # PRINT THE RESULT 
    print("[INFO] Responce: ", results)
          
    # SEND THE RESULT AS JSON OBJECT 
    return jsonify(results[0])

if __name__ == '__main__':
    app.run()