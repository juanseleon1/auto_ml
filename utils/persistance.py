from performance.model_performance import ReportMaker
import joblib
import os
from pandas import DataFrame


class ModelSaver:
    """
    This class is responsible for saving the trained model and generating a report.
    """

    def save_model(self, model, report_dir, data):
        """
        Saves a trained model into a pickle file and generates a report using a ReportMaker object.
    
        Args:
            report_dir (str): The path to the directory where the report will be generated.
            model: The trained model to be saved.
            data: The dataset used to train the model.
    
        Returns:
            None
        """
        filename = os.path.join(report_dir, "pipeline.pkl")
        print(f"Saving best Model to {filename}")
        joblib.dump(model, filename, compress=1)
        reporter = ReportMaker()
        reporter.generate_report(model, data, report_dir)


class ReportWriter:
    """
    This class is responsible for writing the classification report to a CSV file.
    """

    def write_report(self, results, classifier, report_dir):
        """
        Writes the results of a classification model into a CSV file.

        Args:
            results: The results of a classification model.
            classifier (str): The name of the classification model used.
            report_dir (str): The path to the directory where the report will be generated.

        Returns:
            None
        """
        filename = os.path.join(report_dir, f"model_{classifier.named_steps['model']}.csv")
        results_df = DataFrame(data=results)
        results_df.to_csv(filename)
