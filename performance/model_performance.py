from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import os


class ReportMaker:
    """
    A class for generating performance reports for binary classification models.

    Attributes:
        None

    Methods:
        generate_report(model, data, report_dir)
            Generates a report for the given binary classification model using the given dataset.

        calculate_perf_metrics(model, x, y, name, report_dir)
            Calculates and prints the performance metrics for the given binary classification model and dataset.
    """

    def generate_report(self, model, data, report_dir):
        """
        Generates a report for the given binary classification model using the given dataset.

        Args:
            model (sklearn model): The binary classification model to evaluate.
            data (NamedTuple): A named tuple containing the dataset with 'x_train', 'y_train', 'x_test', and 'y_test' fields.
            report_dir (str): The directory to save the report and performance plots.

        Returns:
            None
        """
        print(
            "-------------------------------REPORT FOR TRAIN---------------------------------------"
        )
        self.calculate_perf_metrics(model, data.x_train, data.y_train, "Train", report_dir)
        print(
            "-------------------------------REPORT FOR TEST---------------------------------------"
        )
        self.calculate_perf_metrics(model, data.x_test, data.y_test, "Test", report_dir)

    def calculate_perf_metrics(self, model, x, y, name, report_dir):
        """
        Calculates and prints the performance metrics for the given binary classification model and dataset.

        Args:
            model (sklearn model): The binary classification model to evaluate.
            x (array-like): The features of the dataset.
            y (array-like): The target variable of the dataset.
            name (str): The name of the dataset, either 'Train' or 'Test'.
            report_dir (str): The directory to save the performance plot.

        Returns:
            None
        """
        y_pred = model.predict(x)

        # Calculate the confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        # Calculate the metrics
        accuracy = accuracy_score(y, y_pred)
        sensitivity = recall_score(y, y_pred)
        specificity = tn / (tn + fp)
        precision = precision_score(y, y_pred)

        fpr, tpr, thresholds = roc_curve(y, y_pred)
        auc = roc_auc_score(y, y_pred)
        # Print the results
        print(f"Accuracy: {accuracy:.3f}\n")
        print(f"Sensitivity: {sensitivity:.3f}\n")
        print(f"Specificity: {specificity:.3f}\n")
        print(f"Precision: {precision:.3f}\n")

        print(f"{classification_report(y, y_pred)}")

        plt.plot(fpr, tpr, label=name + " set (AUC = {:.2f})".format(auc))
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        filename = os.path.join(report_dir, f"{name}_roc_curve.png")
        plt.savefig(filename)
