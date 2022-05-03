from wsgiref import simple_server
from flask import Flask, request, render_template, make_response
from flask import Response
import os
from flask_cors import CORS, cross_origin
# import flask_monitoringdashboard as dashboard
import json
import pickle
import pandas as pd
from lightgbm import LGBMRegressor
import joblib
import sklearn
# import xgboost as xgb





os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
#dashboard.bind(app)
#CORS(app)




@app.route("/", methods=['GET'])
# @app.route('/<path:path>')
@cross_origin()
def home():
    # file_path = request.form.get("file_path")
    file=request.files.get("file")
    # file=request.files['csvfile']
    # # print('Home path',file_path)
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():

    try:
        if request.json is not None:
            # Request an input file:-
            print("Getting input file....")
            file = request.files.get("file")

            print("Getting input file....")
            print(type(file))

            input_data_df = pd.read_excel(file)

            # load the Y1 model from disk
            filename_Y1 = 'LGBM_Y1.pickle'
            loaded_model_y1 = pickle.load(open(filename_Y1, 'rb'))

            # load standardizer object from disk
            filename_scaler = 'scaler.pickle'
            loaded_standardizer = pickle.load(open(filename_scaler, 'rb'))

            # Transforming the input
            input_standardized = loaded_standardizer.transform(input_data_df)

            # Prediction
            df_test_pred_y1 = loaded_model_y1.predict(input_standardized)
            df_y1 = pd.DataFrame(df_test_pred_y1, columns=['Y1'])

            # load the Y2 model from disk
            filename_Y2 = 'LGBM_Y2.pickle'
            loaded_model_y2 = pickle.load(open(filename_Y2, 'rb'))

            # Prediction
            df_test_pred_y2 = loaded_model_y2.predict(input_standardized)
            df_y2 = pd.DataFrame(df_test_pred_y2, columns=['Y2'])

            result = pd.concat([input_data_df, df_y1, df_y2], axis=1)

            print("Sending .csv file....")
            resp = make_response(result.to_csv())
            resp.headers["Content-Disposition"] = "attachment; filename=Predictions.csv"
            resp.headers["Content-Type"] = "text/csv"
            result.to_csv("Predictions.csv", header=True, mode='a+')  # appends result to pred

            json_predictions = result.head().to_json(orient="records")
            #
            # return Response("Prediction File created at !!!" + str(path) + 'and few of the predictions are ' + str(
            #     json.loads(json_predictions)))
            return resp

        elif request.form is not None:

            #Request an input file:-
            print("Getting input file....")
            file=request.files.get("file")


            print("Getting input file....")
            print(type(file))

            input_data_df = pd.read_excel(file)

            #load the Y1 model from disk
            filename_Y1 = 'LGBM_Y1.pickle'
            loaded_model_y1 = pickle.load(open(filename_Y1, 'rb'))

            #load standardizer object from disk
            filename_scaler = 'scaler.pickle'
            loaded_standardizer = pickle.load(open(filename_scaler, 'rb'))
            
            #Transforming the input
            input_standardized=loaded_standardizer.transform(input_data_df)
            
            #Prediction
            df_test_pred_y1 = loaded_model_y1.predict(input_standardized)
            df_y1 = pd.DataFrame(df_test_pred_y1, columns=['Y1'])

            #load the Y2 model from disk
            filename_Y2 = 'LGBM_Y2.pickle'
            loaded_model_y2 = pickle.load(open(filename_Y2, 'rb'))
            
            #Prediction               
            df_test_pred_y2 = loaded_model_y2.predict(input_standardized)
            df_y2 = pd.DataFrame(df_test_pred_y2, columns=['Y2'])
            
            result = pd.concat([input_data_df, df_y1, df_y2], axis=1)

            print("Sending .csv file....")
            resp = make_response(result.to_csv())
            resp.headers["Content-Disposition"] = "attachment; filename=Predictions.csv"
            resp.headers["Content-Type"] = "text/csv"
            
            return resp


        else:
        	print("No match found!!")

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


# # @app.route("/return_file", methods=['POST'])
# # @cross_origin()
# def return_file(result_df):

#     try:
#         print("Sending .csv file....")
#         resp = make_response(result_df.to_csv())
#         resp.headers["Content-Disposition"] = "attachment; filename=result.csv"
#         resp.headers["Content-Type"] = "text/csv"
#         return resp

#     except ValueError:
#         return Response("Error Occurred! %s" % ValueError)
#     except KeyError:
#         return Response("Error Occurred! %s" % KeyError)
#     except Exception as e:
#         return Response("Error Occurred! %s" % e)


# port = int(os.getenv("PORT", 8000))
# if __name__ == "__main__":
#     result_df=None
#     app.run(debug=True)

port = int(os.getenv("PORT", 8000))
if __name__ == "__main__":
    host = '0.0.0.0'
    # port = 8000
    httpd = simple_server.make_server(host, port, app)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
    # app.run(host,port, debug=True)
    # docker build -t predictive_maintenence:1.0 .

