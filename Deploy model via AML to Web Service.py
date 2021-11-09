# Databricks notebook source
# MAGIC %md
# MAGIC #Deplpoy ML model via Azure ML
# MAGIC 
# MAGIC *Created by Thorsten Jacobs using DBR 10.0 ML, November 2021.* 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This demo shows how to deploy a ML model trained in Databricks as a web service for real-time scoring. We will use the Iris flower dataset to train a scikit-learn classifier, which we deploy using MLflow and Azure Machine Learning Studio (AML).
# MAGIC 
# MAGIC <img src="https://docs.microsoft.com/en-us/azure/machine-learning/media/how-to-deploy-mlflow-models/mlflow-diagram-deploy.png" width="600"/>
# MAGIC 
# MAGIC The steps are:
# MAGIC - [Set up MLflow](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow) to track model parameters, metrics and artifacts, using AML as [remote tracking server](https://www.mlflow.org/docs/latest/tracking.html#scenario-4-mlflow-with-remote-tracking-server-backend-and-artifact-stores)
# MAGIC - Load the dataset and train the ML model
# MAGIC - Deploy the model to a [web service via AML and the MLflow API](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-mlflow-models)
# MAGIC - Send new data to REST endpoint for [scoring](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?tabs=python)
# MAGIC 
# MAGIC When using this setup all tracking is done in AML studio, note that the experiment will not show up in the Databricks experiment UI.

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

# Susbscription ID for the AML workspace (using secret scope instead of clear text)
subscription_id = dbutils.secrets.get(scope="demo-scope", key="field-eng-subscriptionID")
# Select existing AML workspace or a new name to create a workspace in the resource group
resource_group = 'thortest'
workspace_name = 'aml-thorsten'

# Load or create Azure ML workspace
ws = Workspace.create(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group, exist_ok=True)

# COMMAND ----------

# Configure tracking to AML
import mlflow

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment('iris-sklearn') # Define the experiment name shown in AML here

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
# MAGIC ## Model training

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create additional info to be logged with ML model (optional)
# MAGIC - Pip requirenments file containing the azureml-defaults package, saved on driver node
# MAGIC - Model signature: Schema of input and output data
# MAGIC - Input example data

# COMMAND ----------

# MAGIC %%writefile /tmp/extra_pip_requirements.txt
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

# Create example data to log in MLflow
example_data = data_test.iloc[0:2]

# COMMAND ----------

# MAGIC %md
# MAGIC ###Train model and log with MLflow
# MAGIC 
# MAGIC In this example we use MLflow autologging for parameter and metrics logging, but manually define model parameters logging for customization. When using autologging for models, set the autolog parameter *log_model_signatures=False* when getting a MLflow exception during deployment. 

# COMMAND ----------

from sklearn import svm
mlflow.sklearn.autolog(log_models=False)

artifact_path = 'model' # select folder name for model artifacts
svm_model = svm.SVC()

with mlflow.start_run() as run:
  svm_model.fit(data_train, target_train)
  val_metrics = mlflow.sklearn.eval_and_log_metrics(svm_model, data_test, target_test, prefix="val_")
  
  mlflow.sklearn.log_model(sk_model=svm_model, 
                           artifact_path=artifact_path, 
                           signature=model_signature, 
                           input_example=example_data, 
                           extra_pip_requirements="/tmp/extra_pip_requirements.txt")

# Check results that have been tracked
run_id = run.info.run_id
print(mlflow.get_run(run_id))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Deploy model to web service
# MAGIC 
# MAGIC Log the model in Azure ML Studio and deploy it to a new container instance with REST endpoint. See the docs on how to [update a deployed web service](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-update-web-service). 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 1 - Azure Container Instance (ACI)
# MAGIC 
# MAGIC ACI is recommended for dev/test deployments and low-scale CPU-based workloads.

# COMMAND ----------

# Deployment configuration. Don't specify to use default parameters.
deployment_configs = {
  "computeType": "aci",
  "containerResourceRequirements": {"cpu": 1, "memoryInGB": 1}
}

# COMMAND ----------

from mlflow.deployments import get_deploy_client

# Set the tracking uri as the deployment client
client = get_deploy_client(mlflow.get_tracking_uri())

model_name = "iris-svc-aci" # Select name for model and endpoint
model_uri = "runs:/{}/{}".format(run_id, artifact_path)

# This registers a model to AML without deploying it. You can manually deploy a registered model via the AML UI.
# mlflow.register_model(model_uri=model_uri, name=model_name)

# This registers and deploys a model with standard configurations
deployment_details = client.create_deployment(model_uri=model_uri, config=deployment_configs, name=model_name)

# COMMAND ----------

# Stop execution in case of "run all", to wait for deployment
dbutils.notebook.exit('stop')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 2 - Azure Kubernetes Service (AKS)
# MAGIC 
# MAGIC AKS is recommended for high-scale production workloads. It may take 20-25 minutes to create a new cluster.

# COMMAND ----------

# Step 1 - Create an AKS cluster
from azureml.core.compute import AksCompute, ComputeTarget

# Use the default configuration (can also provide parameters to customize)
prov_config = AksCompute.provisioning_configuration()
aks_name = 'aks-mlflow'

# Create the cluster
aks_target = ComputeTarget.create(workspace=ws, 
                                  name=aks_name, 
                                  provisioning_configuration=prov_config)

aks_target.wait_for_completion(show_output = True)

print(aks_target.provisioning_state)
print(aks_target.provisioning_errors)

# COMMAND ----------

# Deployment configuration
deployment_configs = {
  "computeType": "aks",
  "computeTargetName": "aks-mlflow"
}

# Can also be JSON file
# deployment_configs = {'deploy-config-file': file_path}

# COMMAND ----------

# Step 2 - Register and deploy model with MLflow's deployment client
from mlflow.deployments import get_deploy_client

# Set the tracking uri as the deployment client
client = get_deploy_client(mlflow.get_tracking_uri())

model_name = "iris-svc-aks" # Select name for model and endpoint
model_uri = "runs:/{}/{}".format(run_id, artifact_path)

# the model gets registered automatically and a name is autogenerated using the "name" parameter below 
deployment_details = client.create_deployment(model_uri=model_uri, config=deployment_configs, name=model_name)

# COMMAND ----------

deployment_details

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model scoring - Send request to endpoint
# MAGIC 
# MAGIC See examples for other languages [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?tabs=python).

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

# Get URL for the web service
scoring_uri = deployment_details['scoringUri']
print(scoring_uri)

# COMMAND ----------

# Send request
import requests
import json

# If the service is authenticated, set the key or token
key = ''

# Data to score
data = request_data
# Convert to JSON string
input_data = json.dumps(data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
# headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.text)

# COMMAND ----------

# MAGIC %md
# MAGIC Our endpoint is now life and we can send real-time requests to our model for scoring!
# MAGIC 
# MAGIC Don't forget to **stop the container instance** when it's not needed, for cost control.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up
# MAGIC 
# MAGIC Delete deployed web service

# COMMAND ----------

# Delete AKS instance
aks_target.delete()

# COMMAND ----------

try:
  print(aks_target.get_status())
except:
  print("An exception occurred. The compute target is not found and might have been deleted.")
