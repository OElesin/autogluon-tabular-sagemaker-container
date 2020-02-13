import os
import pickle
import io
import flask
import pandas as pd
import autogluon as ag
from autogluon import TabularPrediction as task
from autogluon.task.tabular_prediction import TabularPredictor
from pandas import DataFrame
import json

model_path = os.environ['MODEL_PATH']


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class AutoGluonTabularService(object):
    """
    Singleton for holding the AutoGluon Tabular task model.
    It has a predict function that does inference based on the model and input data
    """
    model = None

    @classmethod
    def load_model(cls) -> TabularPredictor:
        """Load AutoGluon Tabular task model for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            cls.model = task.load(model_path, verbosity=True)
        return cls.model

    @classmethod
    def predict(cls, prediction_input: DataFrame):
        """For the input, do the predictions and return them.

        Args:
            prediction_input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        prediction_data = task.Dataset(df=prediction_input)
        print("Prediction Data: ")
        print(prediction_data.head())
        return cls.model.predict(prediction_data)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = AutoGluonTabularService.load_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print(f'Request Content Type: {flask.request.content_type}')
    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = io.StringIO(data)
        data = pd.read_csv(s)
    # support for JSON is preferred as it comes with headers
    elif flask.request.content_type == 'application/json':
        raw_payload = flask.request.data.decode('utf-8')
        print(f'Input Data: {raw_payload}')
        payload = json.loads(raw_payload)
        data = pd.DataFrame([payload])
    else:
        return flask.Response(
            response='This predictor only supports JSON or CSV data.  data is preferred.',
            status=415, mimetype='text/plain'
        )

    print('Invoked with {} records'.format(data.shape[0]))
    # Do the prediction
    predictions = AutoGluonTabularService.predict(data)

    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({'results': predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')
