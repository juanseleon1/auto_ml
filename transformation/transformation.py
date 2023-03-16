import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class DataSet:
    """
    A class that splits a pandas DataFrame into training and testing sets.

    Parameters:
    -----------
    frame : pd.DataFrame
        The DataFrame to split into training and testing sets.
    y_label : str
        The name of the column in `frame` to use as the target variable.
    test_size : float, optional
        The proportion of the data to use as the test set. Default is 0.2.
    
    Attributes:
    -----------
    x_train : pd.DataFrame
        The training input data.
    x_test : pd.DataFrame
        The test input data.
    y_train : pd.Series
        The training target variable.
    y_test : pd.Series
        The test target variable.
    """

    def __init__(self, frame, y_label, test_size=0.2) -> None:
        """
        Initializes the DataSet object by splitting the data into training and testing sets.

        Parameters
        ----------
        frame : pd.DataFrame
            The data to split.
        y_label : str
            The name of the target variable.
        test_size : float, optional
            The fraction of the data to use for testing, by default 0.2.
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            frame, frame[y_label], test_size=test_size, random_state=42
        )


class VariableTransformer:
    """
    A class that applies a series of data transformations to a pandas DataFrame.

    Methods:
    --------
    transform(data, y_label, test_size=0.2)
        Applies a series of data transformations to `data` and returns a `DataSet` object.

    set_categories(data)
        Converts specified columns in `data` to categorical data type.

    combine_rare_categories(data, mincount)
        Combines infrequent categories in specified columns of `data` into a new category named 'Other_<column_name>'.

    replace_missing_data(data)
        Replaces missing data in `data` with imputed values.

    create_dummies(data)
        Creates dummy variables for categorical columns in `data`.
    """

    def transform(self, data: pd.DataFrame, y_label, test_size=0.2) -> DataSet:
        """
        Applies a series of data transformations to `data` and returns a `DataSet` object.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame to transform.
        y_label : str
            The name of the column in `data` to use as the target variable.
        test_size : float, optional
            The proportion of the data to use as the test set. Default is 0.2.
        
        Returns:
        --------
        dataset : DataSet
            A `DataSet` object containing the transformed data.
        """
        # TODO: Aqui deberian ir metodos con las transformaciones necesarias.
        print("Transforming the dataset")
        data = self.set_categories(data)
        data = self.combine_rare_categories(data, 10)
        data = self.replace_missing_data(data)
        data = self.create_dummies(data)
        dataset = DataSet(data, y_label, test_size)
        return dataset

    def combine_rare_categories(self, data, mincount):
        """
        Combines categories in categorical columns that occur less than mincount times.

        Parameters
        ----------
        data : pd.DataFrame
            The data to transform.
        mincount : int
            The minimum number of occurrences for a category to be retained.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        for col in data.columns:
            if type(data[col][0]) == str:
                for index, row in pd.DataFrame(data[col].value_counts()).iterrows():
                    if row[0] < mincount:
                        data[col].replace(index, "Other_" + col, inplace=True)
                    else:
                        None
        return data

    def replace_missing_data(self, data):
        """
        Replaces missing data in the input data.

        Args:
            data (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data with the missing data replaced.
        """
        for col in data:
            if data[col].isna().sum() != 0:
                data[col + "_surrogate"] = data[col].isna().astype(int)

            # fixing categoricals
            imputer = SimpleImputer(missing_values=np.nan, strategy="constant")
            imputer.fit(data.select_dtypes(exclude=["int64", "float64"]))
            data[data.select_dtypes(exclude=["int64", "float64"]).columns] = imputer.transform(
                data.select_dtypes(exclude=["int64", "float64"])
            )

            # fixing numericals
            imputer = SimpleImputer(missing_values=np.nan, strategy="median")
            imputer.fit(data.select_dtypes(include=["int64", "float64"]))
            data[data.select_dtypes(include=["int64", "float64"]).columns] = imputer.transform(
                data.select_dtypes(include=["int64", "float64"])
            )
            return data

    def create_dummies(self, data):
        """
        Creates dummy variables for the categorical variables in the input data.

        Args:
            data (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data with the dummy variables created.
        """
        return pd.get_dummies(
            data, columns=data.select_dtypes(exclude=["int64", "float64"]).columns, drop_first=True
        )

    def set_categories(self, data):
        """
        Converts specified columns in `data` to categorical data type.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame to convert.

        Returns:
        --------
        pd.DataFrame
            The converted DataFrame.
        """
        data["From.Grade"] = data["From.Grade"].astype("category")
        data["To.Grade"] = data["To.Grade"].astype("category")
        data["Is.Non.Annual."] = data["Is.Non.Annual."].astype("category")
        data["Parent.Meeting.Flag"] = data["Parent.Meeting.Flag"].astype("category")
        data["Days"] = data["Days"].astype("category")
        data["CRM.Segment"] = data["CRM.Segment"].astype("category")
        data["MDR.High.Grade"] = data["MDR.High.Grade"].astype("category")
        data["School.Sponsor"] = data["School.Sponsor"].astype("category")
        data["NumberOfMeetingswithParents"] = data["CRM.Segment"].astype("category")
        data["SingleGradeTripFlag"] = data["SingleGradeTripFlag"].astype("category")
        return data
