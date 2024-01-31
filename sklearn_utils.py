from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, xai_component

@xai_component
class SKLearnLoadDataset(Component):
    """
    Fetches a specified dataset from sklearn's dataset module.

    #### Reference:
    - [sklearn datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html)

    ##### inPorts:
    - dataset_name: The name of the dataset to be loaded. Provide the name without the 'load_' prefix (e.g., 'iris', 'digits').

    ##### outPorts:
    - dataset: The loaded sklearn dataset, which includes data and target.

    """
    dataset_name: InCompArg[str]
    dataset: OutArg[any]

    def execute(self, ctx) -> None:
        from sklearn import datasets

        # Determine the function name to load the requested dataset
        name = self.dataset_name.value if self.dataset_name.value.startswith("load_") else f"load_{self.dataset_name.value}"
        print(f"Requesting dataset: {self.dataset_name.value}")

        # Attempt to load the dataset
        try:
            load_func = getattr(datasets, name)
            print(f"Loading the '{self.dataset_name.value}' dataset...")
            self.dataset.value = load_func()
            print(f"'{self.dataset_name.value}' dataset loaded successfully.")
        except AttributeError:
            raise ValueError(f"No dataset named '{name}' found in sklearn.datasets")

@xai_component
class SKLearnTrainTestSplit(Component):
    """"
    Takes a sklearn dataset into train and test splits.

    #### Reference:
    - [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

    ##### inPorts:
    - dataset: The input sklearn dataset to be split.
    - train_split: The proportion of the dataset to include in the train split (default is 0.75).
    - random_state: The seed used by the random number generator (default is None).
    - shuffle: Whether or not to shuffle the data before splitting (default is True).
    - stratify: If not None, data is split in a stratified fashion, using this as the class labels (default is None).

    ##### outPorts:
    - X_train: The training data.
    - X_test: The testing data.
    - y_train: The target variable for the training data.
    - y_test: The target variable for the testing data.
    """

    dataset: InCompArg[any]
    train_split: InArg[float]
    random_state: InArg[int]
    shuffle: InArg[bool]
    stratify: InArg[any]
    X_train: OutArg[any] 
    X_test: OutArg[any] 
    y_train: OutArg[any] 
    y_test: OutArg[any] 

    def __init__(self):
        super().__init__()
        self.train_split.value = 0.75
        self.shuffle.value = True

    def execute(self, ctx) -> None:
        
        from sklearn.model_selection import train_test_split

        print(f"Split Parameters:\nTrain Split {self.train_split.value} \nShuffle: {self.shuffle.value} \nRandom State: {self.random_state.value}")
        self.X_train.value, self.X_test.value, self.y_train.value, self.y_test.value = train_test_split(self.dataset.value['data'], self.dataset.value['target'], 
                                    test_size=self.train_split.value, shuffle=self.shuffle.value, 
                                    random_state=self.random_state.value, stratify=self.stratify.value)
        
        print(f"Train data shape: {self.X_train.value.shape}, Train target shape: {self.y_train.value.shape}")
        print(f"Test data shape: {self.X_test.value.shape}, Test target shape: {self.y_test.value.shape}")


@xai_component
class CSVToSKLearnDataset(Component):
    """
    Transforms a CSV file into a format compatible with sklearn.datasets.

    This component reads a CSV file, selects specific columns to use as features and a target, converts it to a pandas DataFrame, and then transforms the DataFrame into a format compatible with sklearn.datasets. If the target column is categorical, it will be label encoded to numerical values.

    #### Reference:
    - [Pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
    - [sklearn.datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html)
    - [sklearn LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

    ##### inPorts:
    - file_path: The path to the CSV file to be transformed.
    - feature_columns: The list of columns in the CSV file to use as data. If not specified, all columns except the target will be used.
    - target_column: The column in the CSV file to use as the target variable.
    - drop_na_rows: If set to True, rows with any NA/missing values will be dropped. Defaults to False.

    ##### outPorts:
    - dataset: The sklearn compatible dataset.
    """
    file_path: InArg[str]
    feature_columns: InArg[list]
    target_column: InArg[str]
    drop_na_rows: InArg[bool]
    dataset: OutArg[dict]

    def __init__(self):
        super().__init__()
        self.drop_na_rows.value = False
    
    def execute(self, ctx) -> None:
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder

        df = pd.read_csv(self.file_path.value)

        # If feature_columns are not provided, use all columns except the target_column
        if not self.feature_columns.value:
            self.feature_columns.value = df.columns.drop(self.target_column.value).tolist()

        # Use only selected columns as features
        df = df[self.feature_columns.value + [self.target_column.value]]

        # If drop_na_rows is True, drop any rows with NA/missing values
        if self.drop_na_rows.value:
            df = df.dropna()

        target = df.pop(self.target_column.value)
        # If the target data type is object (likely strings), label encode it
        if target.dtype == 'object':
            le = LabelEncoder()
            target = le.fit_transform(target)

        data = df.values

        self.dataset.value = {
            'data': data,
            'target': target,
            'feature_names': df.columns.tolist(),
            'DESCR': f'Dataset loaded from {self.file_path.value}, target column is {self.target_column.value}'
        }
        
        print(f"Data shape: {data.shape}, Target shape: {target.shape}")

@xai_component
class SKLearnModelTraining(Component):
    """
    Trains a specified scikit-learn model using the provided training data.

    #### Reference:
    - [sklearn estimators](https://scikit-learn.org/stable/user_guide.html)

    ##### inPorts:
    - X_train: Training data features.
    - y_train: Training data targets.
    - model: The scikit-learn model to train.

    ##### outPorts:
    - trained_model: The trained scikit-learn model.
    """

    X_train: InCompArg[any]
    y_train: InCompArg[any]
    model: InCompArg[any]
    trained_model: OutArg[any]

    def execute(self, ctx) -> None:
        print("Training model...")
        self.trained_model.value = self.model.value.fit(self.X_train.value, self.y_train.value)
        print("Training complete.")


@xai_component
class SKLearnClassificationEvaluation(Component):
    """
    Evaluates a trained scikit-learn classification model using testing data, providing key metrics such as accuracy, precision, recall, and F1 score.

    #### Reference:
    - [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)

    ##### inPorts:
    - X_test: Testing data features.
    - y_test: Testing data targets.
    - trained_model: The trained scikit-learn classification model.
    - average_method: The averaging method for multi-class classification metrics ('micro', 'macro', 'weighted'). Default is 'macro'.

    ##### outPorts:
    - evaluation_metrics: The performance metrics of the model on testing data.
    """

    X_test: InCompArg[any]
    y_test: InCompArg[any]
    trained_model: InCompArg[any]
    average_method: InArg[str] = 'macro'  # Set default value directly here
    evaluation_metrics: OutArg[dict]

    def execute(self, ctx) -> None:
        from sklearn import metrics

        print("Evaluating classification model...")
        predictions = self.trained_model.value.predict(self.X_test.value)
        accuracy = metrics.accuracy_score(self.y_test.value, predictions)
        precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(
            self.y_test.value, predictions, average=self.average_method.value
        )

        self.evaluation_metrics.value = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

        print("\nEvaluation Metrics:")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision.mean():.4f} (Average)")
        print(f"Recall   : {recall.mean():.4f} (Average)")
        print(f"F1 Score : {f1_score.mean():.4f} (Average)")