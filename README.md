# SKLearn XAI Components Library

This library provides a collection of components for working with scikit-learn models, datasets, and evaluation tools within the XAI framework. It includes components for loading datasets, data preprocessing, model training, evaluation, and various machine learning algorithms.

## Prerequisites

- Python 3.8 or higher
- scikit-learn
- pandas (for CSV data handling)

## Installation

You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```

To use this component library, simply copy the directory / clone or submodule the repository to your working Xircuits project directory.

## Components Overview

The library includes a variety of components categorized into dataset handling, data preprocessing, model training, and model evaluation.

### Dataset Handling

- **`SKLearnLoadDataset`**: Fetches datasets from scikit-learn's dataset collection.
- **`CSVToSKLearnDataset`**: Converts a CSV file into a format compatible with scikit-learn datasets.

### Data Preprocessing

- **`SKLearnTrainTestSplit`**: Splits datasets into training and testing sets.

### Model Training

- **`SKLearnModelTraining`**: Trains a specified scikit-learn model using provided training data.
- **`SKLearnRandomForestClassifier`**: Initializes a RandomForestClassifier model.
- **`SKLearnLogisticRegression`**: Initializes a LogisticRegression model.
- **`SKLearnSVC`**: Initializes a Support Vector Classifier (SVC) model.
- **`SKLearnKNeighborsClassifier`**: Initializes a KNeighborsClassifier model.
- **`SKLearnDecisionTreeClassifier`**: Initializes a DecisionTreeClassifier model.
- **`SKLearnGradientBoostingClassifier`**: Initializes a GradientBoostingClassifier model.
- **`SKLearnSVR`**: Initializes a Support Vector Regression (SVR) model.
- **`SKLearnMultinomialNB`**: Initializes a Multinomial Naive Bayes (MultinomialNB) model.
- **`SKLearnRidgeRegression`**: Initializes a Ridge Regression model.
- **`SKLearnKMeans`**: Initializes a KMeans clustering model.

### Model Evaluation

- **`SKLearnClassificationEvaluation`**: Evaluates a trained scikit-learn classification model using testing data.

## Usage

Each component can be integrated into your XAI workflows as needed. For instance, to train a RandomForestClassifier model:

1. Load your dataset using `SKLearnLoadDataset` or `CSVToSKLearnDataset`.
2. Split the dataset into training and testing sets with `SKLearnTrainTestSplit`.
3. Initialize the RandomForestClassifier model using `SKLearnRandomForestClassifier`.
4. Train the model with `SKLearnModelTraining` using the training data.
5. Evaluate the model's performance on the test set using `SKLearnClassificationEvaluation`.

Refer to the component documentation for detailed usage instructions and parameter explanations.

## Contributing

We welcome contributions to this library. If you have suggestions for new components or improvements to existing ones, please open an issue or submit a pull request.
