# Databricks notebook source
# MAGIC %md
# MAGIC #Deplpoy ML model via Azure ML
# MAGIC 
# MAGIC *Created by Thorsten Jacobs using DBR 10.0 ML, November 2021.* 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In this demo we will train a simple ML model in Databricks and deploy it as web service for live scoring, using MLflow and Azure Machine Learning Studio (AML). We will use the Iris flower dataset and build a support vector classifier with scikit-learn.
# MAGIC 
# MAGIC <img src="https://docs.microsoft.com/en-us/azure/machine-learning/media/how-to-deploy-mlflow-models/mlflow-diagram-deploy.png" width="600"/>
# MAGIC 
# MAGIC The steps are:
# MAGIC - [Set up MLflow](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow) to track model parameters, metrics and artifacts, using AML as [remote tracking server](https://www.mlflow.org/docs/latest/tracking.html#scenario-4-mlflow-with-remote-tracking-server-backend-and-artifact-stores)
# MAGIC - Load the dataset and train the ML model
# MAGIC - Deploy the model to a [web service via AML and the MLflow API](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-mlflow-models)
# MAGIC - Send new data to REST endpoint for [scoring](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?tabs=python) 

# COMMAND ----------

# MAGIC %md
# MAGIC **Prerequisites:**
# MAGIC - Use a recent ML runtime or install mlflow manually.
# MAGIC - If you want to use an existing Azure ML instance, make sure you have a [user role](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-assign-roles) with the required access rights in AML.
# MAGIC - This example uses interactive authentication to AML. Use a [service principal](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication#configure-a-service-principal) for production code.
# MAGIC 
# MAGIC We will use the azureml-mlflow package, which we deploy in the next step

# COMMAND ----------

# MAGIC %pip install azureml-mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ##Set up mlflow tracking to AML

# COMMAND ----------

# Connect to or create Azure ML studio workspace
from azureml.core import Workspace

# Susbscription ID and resource group of the AML workspace 
subscription_id = '3f2e4d32-8e8d-46d6-82bc-5bb8d962328b'
resource_group = 'thortest'
# Select existing AML workspace or a new name to create a workspace 
workspace_name = 'aml-thorsten'

# Load or create Azure ML workspace
ws = Workspace.create(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group, exist_ok=True)

# COMMAND ----------

# Configure tracking to AML
import mlflow

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment('ADB-AML-iris') # Define the experiment name shown in AML here

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and prep data

# COMMAND ----------

# Load dataset and split to train and test set
import pandas as pd
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_iris()
train = pd.DataFrame(data.data, columns=data.feature_names)

data_train, data_test, target_train, target_test = train_test_split(train, data.target)

data_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train ML model

# COMMAND ----------

# MAGIC %md
# MAGIC ** Create additional info to be logged with ML model (optional)**
# MAGIC - Pip requirenments file containing the azureml-defaults package, saved on driver node
# MAGIC - Model signature: Schema of input and output data
# MAGIC - Input example data

# COMMAND ----------

# MAGIC %%writefile ./extra_pip_requirements.txt
# MAGIC azureml-defaults

# COMMAND ----------

# Manually specify model signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

input_schema = Schema([
  ColSpec("double", "sepal length (cm)"),
  ColSpec("double", "sepal width (cm)"),
  ColSpec("double", "petal length (cm)"),
  ColSpec("double", "petal width (cm)"),
])
output_schema = Schema([ColSpec("integer")])
model_signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Create example data
example_data = data_test.iloc[0:2]

# COMMAND ----------

# Train model and log with mlflow
from sklearn import svm
from mlflow.models.signature import infer_signature

# Log training metrics with MLflow autologging
# Disable autologging for model artifact and use manual model logging instead
mlflow.sklearn.autolog(log_model_signatures=False, log_models=False)

artifact_path = 'model' # select folder name for model artifacts

svm_model = svm.SVC()
with mlflow.start_run() as run:
  svm_model.fit(data_train, target_train)
  val_metrics = mlflow.sklearn.eval_and_log_metrics(svm_model, data_test, target_test, prefix="val_")
  
  mlflow.sklearn.log_model(sk_model=svm_model, 
                           artifact_path=artifact_path, 
                           signature=model_signature, 
                           input_example=example_data, 
                           extra_pip_requirements="extra_pip_requirements.txt")

# Check results that have been tracked
run_id = run.info.run_id
print(mlflow.get_run(run_id))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Deploy model via Azure ML

# COMMAND ----------

# DBTITLE 1,Register and deploy model
from mlflow.deployments import get_deploy_client

# Set the tracking uri as the deployment client
client = get_deploy_client(mlflow.get_tracking_uri())

model_name = "iris-svc" # select model name shown in AML
model_uri = "runs:/{}/{}".format(run_id, artifact_path)

# This registers a model to AML without deploying it. You can manually deploy a registered model via the AML UI.
# mlflow.register_model(model_uri=model_uri, name=model_name)

# This registers and deploys a model with standard configurations
client.create_deployment(model_uri=model_uri, name=model_name)

# COMMAND ----------

# Stop execution in case of "run all" to wait for deployment
dbutils.notebook.exit('stop')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Send request to endpoint
# MAGIC 
# MAGIC Once the endpoint is deployed we can send new data to the REST API for scoring. Insert the REST endpoint URL and API key (if activated) below. You can find them under 'Endpoints' in Azure ML studio, toghether with example code for other languages (under 'Consume').

# COMMAND ----------

# Format new data in JSON structure to send to API
new_data = data_test[:2]
request_data = {
  "input_data":{
    "columns": new_data.columns.tolist(),
    "index": new_data.index.tolist(),
    "data": new_data.values.tolist()
  }
}
print(request_data)

# COMMAND ----------

import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
data = request_data

body = str.encode(json.dumps(data))

url = 'http://20031300-c458-423b-826e-2d767faef75f.northeurope.azurecontainer.io/score'
api_key = '' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))
