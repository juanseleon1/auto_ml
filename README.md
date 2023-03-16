# Auto ML Library

Auto ML is a Python library that provides an easy-to-use interface for automating machine learning workflows. This library includes tools for data preprocessing, feature engineering, model selection, hyperparameter tuning, and model evaluation. It is built with the goal of making machine learning accessible to a wider range of users by reducing the time and effort required for training and deploying models.

## Installation 
Create and activate a virtual environment:
Use python 3.8 o >
On Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows CMD :
```bash
py -m venv .venv
.venv\Scripts\activate.bat
```
On Windows PowerShell :
```bash
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Then inside the root folder run 
```bash
pip install -e .
```
## Usage
### start_auto_ml Command
```bash
usage: start_auto_ml [-h] [--models {svm,rf,bayes,knn} [{svm,rf,bayes,knn} ...]] [--score_func {accuracy,f1,recall,auc,precision}] [--threads {1..6}] [--test_size TEST_SIZE] [--folds {2..10}] [--train_report] [--class_col CLASS_COL] datafile report_name

This program does AutoML on a given dataset

positional arguments:
  datafile              Input data file name (mandatory)
  report_name           Report name (mandatory)

optional arguments:
  -h, --help            show this help message and exit
  --models {svm,rf,bayes,knn} [{svm,rf,bayes,knn} ...]
                        List of models to use (not mandatory, default: [svm, knn, bayes])
  --score_func {accuracy,f1,recall,auc,precision}
                        Scoring function to use (not mandatory)
  --threads {1..6}      Number of threads to use (not mandatory)
  --test_size TEST_SIZE
                        Percentage of data to use as test (not mandatory)
  --folds {2..10}       Number of folds to use (not mandatory)
  --train_report        Whether to generate a training report (not mandatory)
  --class_col CLASS_COL
                        Classes that are objectives(not mandatory)
```
#### Parameters
datafile (mandatory)
The input data file name.

report_name (mandatory)
The name of the report to be generated.

--models (not mandatory)
List of models to use. The default models are svm, knn, and bayes.

--score_func (not mandatory)
The scoring function to use. The default scoring function is accuracy.

--threads (not mandatory)
The number of threads to use. The default value is 2.

--test_size (not mandatory)
The percentage of data to use as test. The default value is 0.2.

--folds (not mandatory)
The number of folds to use. The default value is 4.

--train_report (not mandatory)
Whether to generate a training report. By default, this is set to False.

--class_col (not mandatory)
The classes that are objectives. The default value is "Retained.in.2012.".
### generate_eda Command
```bash
generate_eda datafile
```
## Model evaluation
Auto ML includes tools for evaluating the performance of models, including:

Confusion matrix: Calculate the confusion matrix for the model.
Accuracy: Calculate the accuracy of the model.
Precision: Calculate the precision of the model.
Recall: Calculate the recall of the model.
ROC curve: Plot the ROC curve for the model.