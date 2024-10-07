
<p align="center">
  <a href="https://github.com/XpressAI/xircuits/tree/master/xai_components#xircuits-component-library-list">Component Libraries</a> •
  <a href="https://github.com/XpressAI/xircuits/tree/master/project-templates#xircuits-project-templates-list">Project Templates</a>
  <br>
  <a href="https://xircuits.io/">Docs</a> •
  <a href="https://xircuits.io/docs/Installation">Install</a> •
  <a href="https://xircuits.io/docs/category/tutorials">Tutorials</a> •
  <a href="https://xircuits.io/docs/category/developer-guide">Developer Guides</a> •
  <a href="https://github.com/XpressAI/xircuits/blob/master/CONTRIBUTING.md">Contribute</a> •
  <a href="https://www.xpress.ai/blog/">Blog</a> •
  <a href="https://discord.com/invite/vgEg2ZtxCw">Discord</a>
  
<p align="center">
<img src= https://github.com/user-attachments/assets/2eb68c60-d3b6-4209-8297-903874db8ab5
" width="450"/>
</p>

<p align="center"><i>Xircuits Component Library for integrating scikit-learn models, datasets, and evaluation tools.</i></p>

---

## Welcome to the SKLearn XAI Components Library

The **SKLearn XAI Components Library** provides a simple and intuitive way to integrate **scikit-learn** machine learning models, datasets, and evaluation tools within the **XAI**  framework. With this library, you can easily manage the end-to-end workflow for data handling, model training, and evaluation using scikit-learn's extensive set of algorithms and features.

In this guide, you will find the steps to install the library, set up a workflow, and get started with training and evaluating machine learning models.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started with SKLearn XAI Components](#getting-started-with-sklearn-xai-components)
  - [Example Workflow: RandomForestClassifier](#example-workflow-randomforestclassifier)
- [Components Overview](#components-overview)
  - [Dataset Handling](#dataset-handling)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)

## Prerequisites

Before using this library, you’ll need the following:

1. **Python 3.8** or higher
2. **scikit-learn**: Core machine learning algorithms
3. **pandas**: For handling CSV and tabular data
4. **Xircuits**: To integrate the components into your workflows

## Installation

To install the SKLearn XAI Components Library, follow these steps:

### 1. Install the required dependencies

Ensure that the required libraries are installed by running:

```bash
pip install -r requirements.txt
```

### 2. Add the components to your Xircuits project

You can either pull the repository as a submodule or manually clone it into your **Xircuits** project directory:

```bash
# To clone the repository into your Xircuits project directory
git clone https://github.com/XpressAI/xai-sklearn.git xai_components/xai_sklearn

# Install required dependencies
pip install -r xai_components/xai_sklearn/requirements.txt
```

## Getting Started with SKLearn XAI Components

Once you have installed the required libraries and components, you can start building machine learning workflows in **Xircuits** using the **SKLearn XAI Components Library**.

Below is a sample workflow for training and evaluating a **RandomForestClassifier**.

### Example Workflow: RandomForestClassifier

1. **Load a Dataset**  
   Use `SKLearnLoadDataset` to load built-in datasets such as **Iris**:
   ```python
   dataset = SKLearnLoadDataset('iris')
   ```

2. **Split the Dataset**  
   Split the data into training and testing sets using `SKLearnTrainTestSplit`:
   ```python
   X_train, X_test, y_train, y_test = SKLearnTrainTestSplit(dataset)
   ```

3. **Initialize and Train the Model**  
   Initialize and train a **RandomForestClassifier**:
   ```python
   model = SKLearnRandomForestClassifier()
   trained_model = SKLearnModelTraining(model, X_train, y_train)
   ```

4. **Evaluate the Model**  
   Evaluate the model's performance using `SKLearnClassificationEvaluation`:
   ```python
   SKLearnClassificationEvaluation(trained_model, X_test, y_test)
   ```

For more detailed examples and component usage, refer to the **Component Documentation**.

## Components Overview

The **SKLearn XAI Components Library** is structured into several categories to cover the entire workflow of dataset handling, model training, and evaluation.

### Dataset Handling

- **`SKLearnLoadDataset`**: Fetches built-in datasets from scikit-learn.
- **`CSVToSKLearnDataset`**: Converts a CSV file into a format compatible with scikit-learn datasets.

### Data Preprocessing

- **`SKLearnTrainTestSplit`**: Splits datasets into training and testing sets with configurable options like shuffle and random state.

### Model Training

- **`SKLearnModelTraining`**: Trains a scikit-learn model with the provided training data.
- **`SKLearnRandomForestClassifier`**: Initializes a **RandomForestClassifier**.
- **`SKLearnLogisticRegression`**: Initializes a **LogisticRegression** model.
- **`SKLearnSVC`**: Initializes a **Support Vector Classifier (SVC)**.
- **`SKLearnKNeighborsClassifier`**: Initializes a **KNeighborsClassifier**.
- **`SKLearnDecisionTreeClassifier`**: Initializes a **DecisionTreeClassifier**.
- **`SKLearnGradientBoostingClassifier`**: Initializes a **GradientBoostingClassifier**.
- **`SKLearnSVR`**: Initializes a **Support Vector Regression (SVR)** model.
- **`SKLearnMultinomialNB`**: Initializes a **Multinomial Naive Bayes** model.
- **`SKLearnRidgeRegression`**: Initializes a **Ridge Regression** model.
- **`SKLearnKMeans`**: Initializes a **KMeans** clustering model.

### Model Evaluation

- **`SKLearnClassificationEvaluation`**: Evaluates a trained classification model using metrics such as accuracy, precision, recall, and F1-score.

## Contributing

We welcome contributions to the **SKLearn XAI Components Library**! If you would like to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Open a pull request with a detailed description of your changes.

Please feel free to suggest new components, improvements, or optimizations. If you encounter any issues or have ideas for enhancements, you can open an issue in the repository.

---

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

