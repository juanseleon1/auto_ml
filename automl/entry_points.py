from automl.maker import AutoMLMaker
from argparse import ArgumentParser, Namespace


def start_auto_ml():
    """
    The start_auto_ml function takes in user input arguments using the ArgumentParser module to perform Auto Machine Learning on a given dataset.

    Args:
    None
    
    Returns:
    None
    
    Raises:
    None
    """
    parser = ArgumentParser(
        prog="start_auto_ml", description="This program does auto ML on a Given dataset"
    )
    parser.add_argument("datafile", type=str, help="Input data file name (mandatory)")
    parser.add_argument("--report_name", type=str, help="Report name (mandatory)")

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["svm", "rf", "bayes", "knn"],
        default=["svm", "knn", "bayes"],
        help="List of models to use (not mandatory, default: [svm, lr, bayes])",
    )
    parser.add_argument(
        "--score_func",
        type=str,
        choices=["accuracy", "f1", "recall", "precision"],
        default="accuracy",
        help="Scoring function to use (not mandatory)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        choices=range(1, 7),
        default=2,
        help="Number of threads to use (not mandatory)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Percentage of data to use as test (not mandatory)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        choices=range(2, 11),
        default=4,
        help="Number of folds to use (not mandatory)",
    )
    parser.add_argument(
        "--train_report",
        default=False,
        action="store_true",
        help="Whether to generate a training report (not mandatory)",
    )
    parser.add_argument(
        "--class_col",
        type=str,
        default="Retained.in.2012.",
        help="Classes that are objectives(not mandatory)",
    )

    args: Namespace = parser.parse_args()
    print("Starting AutoML")
    auto_ml = AutoMLMaker(args.datafile, args.report_name, args.class_col)
    auto_ml.start_auto_ml(
        args.models, args.score_func, args.threads, args.folds, args.train_report, args.test_size
    )
