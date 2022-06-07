###Imports
import pickle
import pandas as pd
from flask            import Flask, request, Response
from rossmann.Rossmann import Rossman

###Loading model
model = pickle.load(open('/media/luanzitto/Install/Luan-PC/Documents/01-Estudos/01-Cursos/01- Comunidade_DS/01-Data_Science_Producao/models/model_xgboost.pkl', 'rb'))

###Initialize API
app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])

###Functions
def rossmann_predict():
    test_json = request.get_json()

    if test_json: #there is data
        if isinstance(test_json, dict): #unique example
            test_raw = pd.DataFrame(test_json, index=[0])
        else: #multiple examples
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        #Instantiate Rossmann class
        pipeline = Rossman()

        
        #Data cleaning
        df = pipeline.data_cleaning(test_raw)
        #Feature engineering
        df1 = pipeline.feaure_engineering(df)
        #Data preparation
        df2 = pipeline.data_preparation(df1)
        #Prediction
        df_response = pipeline.get_prediction(model, test_raw, df2)

        return df_response


    else:
        return Response('{}', status=200, mimetype='application/json')            

if __name__ == '__main__':
    app.run('0.0.0.0')