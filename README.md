# Informations

The original code is available at: https://github.com/dionatrafk/model_evaluation?tab=readme-ov-file



This project is an updated version for python3.

The folders:
-  **http_requests_nasa** 

Contains the NASA, MIT and Alibaba datasets corresponding to the prediction intervals. For example, trace60.csv contains the data needed to make predictions for the next 60 minutes.

Original dataset:
 - NASA: http://ita.ee.lbl.gov/html/contrib/
 - MIT: https://registry.opendata.aws/dcc/
 - Alibaba: https://github.com/alibaba/clusterdata


- **algorithms**
  
Contains the source codes of Machine Learning models (MLP, GRU, ARIMA) and their hyperparameter optimizers (*Grid Search* (**grid_search.py**), *Tree of parzen estimators (TPE)* (**hypeas_gru .py** and **hypeas_mlp.py**) for local execution.

To execute the algorithms, you will need an environment with python version 3.11.
Python 3.11 available at: https://www.python.org/downloads/release/python-3119/

To create an environment, run:
py -3.11 -m venv .venv

Activate the environment:
.\.venv\Scripts\Activate.ps1

And install the packages:
pip install -r requirements.txt

The algorithms below are available as an alternative for online execution:
- **MLP_exec.ipynb**
- **GRU_exec.ipynb**
- **ARIMA_exec.ipynb**

They correspond to the source code of each Machine Learning model. They are in a similar format to the Jupyter notebook.
When accessing the file page, select *Open in Colab* to access the model in execution mode in the *Google Colaboratory* environment.
