from datetime import datetime
import os
from ydata_profiling import ProfileReport
import pandas as pd


class EDAMaker:
    """
    A class that generates Exploratory Data Analysis (EDA) reports.

    Attributes:
        data (pandas.DataFrame or None): The dataset to perform EDA on.

    Methods:
        generate_eda_report(data, report_dir=None)
    """

    def __init__(self) -> None:
        """
        Initializes an EDAMaker object.
        """
        self.data = None

    def generate_eda_report(self, data, report_dir=None):
        """
        Generates an EDA report for the given dataset.

        Args:
            data (str or pandas.DataFrame): The path to the dataset file or the dataset itself.
            report_dir (str): The directory to save the report to. If not provided, saves to the current working directory.

        Returns:
            None
        """
        if not report_dir:
            report_dir = os.getcwd()
        if isinstance(data, str):
            self.data = pd.read_csv(data, header=0)
        else:
            self.data = data
        date_file_name = "ProfileReport_{}.html".format(datetime.now().strftime("%Y-%m-%d"))
        filename = os.path.join(report_dir, date_file_name)
        profile = ProfileReport(
            self.data, title=f"Profiling Report", html={"style": {"full_width": True}}
        )
        profile.to_file(output_file=filename)
