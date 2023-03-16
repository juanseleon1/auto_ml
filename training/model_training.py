from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils.persistance import ReportWriter


class Trainer:
    """
    This class provides a framework for training different models on a given dataset and searching for the best model
    based on the specified hyperparameters and scoring metric.

    Attributes:
    max_score (float): The highest score achieved by any model during training.
    to_be_ran (list): A list of the names of the models that are going to be trained.
    scoring (str): The scoring metric used to evaluate the models.
    threads (int): The number of threads to use for training the models.
    splits (int): The number of splits to use for cross-validation.
    generate_train_report (bool): A flag to determine whether or not to generate a training report.
    param_map (dict): A dictionary of hyperparameter search spaces for each model.
    model_map (dict): A dictionary of pipeline objects for each model.
    """

    def __init__(
        self, models, params, scoring="accuracy", threads=-1, splits=10, generate_train_report=False
    ) -> None:
        """
        Initializes the Trainer class with the specified hyperparameters.
        Args:
            models (list): A list of the names of the models to be trained.
            params (dict): A dictionary of hyperparameter search spaces for each model.
            scoring (str): The scoring metric used to evaluate the models. Default is "accuracy".
            threads (int): The number of threads to use for training the models. Default is -1 (use all available CPUs).
            splits (int): The number of splits to use for cross-validation. Default is 10.
            generate_train_report (bool): A flag to determine whether or not to generate a training report. Default is False.
        """
        seed = 42
        self.max_score = 0
        self.to_be_ran = models
        self.scoring = scoring
        self.threads = threads
        self.splits = splits
        self.generate_train_report = generate_train_report
        self.param_map = params
        self.model_map = {
            "svm": Pipeline(
                [("scaler", StandardScaler()), ("model", SVC(probability=True, random_state=seed))]
            ),
            "rf": Pipeline(
                [("scaler", StandardScaler()), ("model", RandomForestClassifier(random_state=seed))]
            ),
            "bayes": Pipeline([("scaler", StandardScaler()), ("model", GaussianNB())]),
            "knn": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
        }

    def search_models(self, data, report_dir):
        """
        Searches the specified models using the specified hyperparameters and scoring metric on the provided data.
        Returns the best model based on the highest achieved score.

        Args:
            data (Data): An instance of the Data class containing the data to be used for training and testing.
            report_dir (str): The directory in which to write the training report (if generate_train_report is True).

        Returns:
            The best estimator of the trained model.
        """
        for model in self.to_be_ran:
            print(f"Training {model}")
            score, best_clf = self.train_model(
                self.model_map[model], self.param_map[model], data, report_dir
            )
            if score > self.max_score:
                self.max_score = score
                self.best_model = best_clf
        print(f"Best Model is {self.best_model.named_steps['model']}")
        return self.best_model

    def train_model(self, classifier, params, data, report_dir):
        """
        Trains a given classifier with the specified parameters on the provided data using cross-validation. 
        Returns the best trained classifier along with its corresponding score.
    
        Parameters:
        - classifier (sklearn Pipeline): The classifier to train
        - params (dict): Dictionary of hyperparameters for the classifier
        - data (DataHandler): An instance of DataHandler containing the training data
        - report_dir (str): Directory to save the training report
    
        Returns:
        - Tuple: A tuple containing the best score and the corresponding best trained classifier.
        """
        print(f"Training classifiers for: {classifier.named_steps['model']}")
        clf = GridSearchCV(
            classifier,
            param_grid=params,
            cv=StratifiedKFold(n_splits=self.splits),
            scoring=self.scoring,
            n_jobs=self.threads,
            refit=True,
            verbose=0,
            return_train_score=True,
        )
        clf.fit(data.x_train, data.y_train)
        print(f"For model {classifier.named_steps['model']}")
        print(f"Best score {clf.best_score_}")
        if self.generate_train_report:
            print(f"Writing model report")
            ReportWriter().write_report(clf.cv_results_, classifier, report_dir)
        return clf.best_score_, clf.best_estimator_
