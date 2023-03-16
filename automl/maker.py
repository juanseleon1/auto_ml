import pandas as pd
from eda.eda import EDAMaker
from training.model_training import Trainer
from transformation.transformation import VariableTransformer, DataSet
from utils.param_reader import read_params
from utils.persistance import ModelSaver
import errno
import os
from datetime import datetime

curr_dir = os.getcwd()


class AutoMLMaker:
    """
    A class that performs Automated Machine Learning (AutoML) on a given dataset.

    Args:
        datafile (str): Name of the input data file.
        report_name (str): Name of the report file to be generated.
        class_col (str): Name of the column that contains the objective classes.

    Attributes:
        datafile (str): Name of the input data file.
        report_name (str): Name of the report file to be generated.
        class_col (str): Name of the column that contains the objective classes.
    """

    def __init__(self, file_path, report_name, class_col) -> None:
        """The __init__ method initializes an instance of the AutoMLMaker class.

        Args:
        file_path (str): Path to the input data file.
        report_name (str): The name of the report to be generated.
        class_col (str): The name of the column in the input data file that represents the classes.

        Returns:
        None.
        """
        self.data = pd.read_csv(file_path, header=0)
        self.report_name = report_name
        self.class_col = class_col

    def start_auto_ml(self, to_run, scoring, threads, split, report, test_size=0.2):
        """The start_auto_ml method starts the auto-ml process.

        Args:
        to_run (list): A list of machine learning models to run.
        scoring (str): The scoring metric to use.
        threads (int): The number of threads to use for parallel processing.
        split (int): The number of folds to use for cross-validation.
        report (bool): A flag indicating whether to generate a training report.
        test_size (float): The percentage of data to use as the test set.

        Returns:
        None.
        """
        report_dir = self.determine_report_dir()
        params = read_params()
        print("AutoML parameters loaded successfully")
        if report:
            print("Generating EDA report")
            eda = EDAMaker()
            eda.generate_eda_report(self.data, report_dir)
        transformer = VariableTransformer()
        dataset = transformer.transform(self.data, self.class_col, test_size)
        trainer = Trainer(to_run, params, scoring, threads, split, report)
        model = trainer.search_models(dataset, report_dir)
        ModelSaver().save_model(model, report_dir, dataset)

    def determine_report_dir(self):
        """The determine_report_dir method determines the report directory.

        Args:
        None.

        Returns:
        report_dir (str): The path to the directory where the report will be generated.

        Returns:
        """
        folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if self.report_name:
            folder_name = self.report_name
        report_dir = os.path.join(curr_dir, folder_name)
        if not os.path.exists(report_dir):
            try:
                os.makedirs(report_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise  # This was not a "directory exist" error..
        return report_dir
