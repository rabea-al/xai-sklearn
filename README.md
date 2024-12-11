



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
</p>

<p align="center">
<img src= https://github.com/user-attachments/assets/2eb68c60-d3b6-4209-8297-903874db8ab5
" width="450"/>
</p>

<p align="center"><i>Xircuits Component Library for integrating scikit-learn models, datasets, and evaluation tools.</i></p>

---

## Table of Contents

- [Preview](#preview)  
- [Prerequisites](#prerequisites)  
- [Main Components Library](#main-components-library)   
- [Try the Examples](#try-the-examples)  
- [Installation](#installation)  

## Xircuits Component Library for SkLearn
This library enables seamless integration of scikit-learn's machine learning models, datasets, and evaluation tools into Xircuits, streamlining data workflows, model training, and performance evaluation.

## Preview

### The Example:
![sklearn_example](https://github.com/user-attachments/assets/565a4919-8b67-4ced-9ad5-47a2645bf3c6)

### The Result:
![sklearn_result](https://github.com/user-attachments/assets/c2bc95b4-41f9-4d8d-a1a3-9d6a9097d3c9)

## Prerequisites

Before you begin, you will need the following:

1. Python3.9+.
2. Xircuits.

## Main Components Library

### SKLearnRandomForestClassifier Component:
Initializes a RandomForestClassifier for high-accuracy classification tasks, using specified or default parameters.

<p align="center"><img src="https://github.com/user-attachments/assets/93489276-7c1d-4db1-ab9c-25ca6b027f0b" alt="SKLearnRandomForestClassifier" width="200" height="75" />

#### SKLearnLogisticRegression Component:  
Initializes a LogisticRegression model, widely used for binary classification and multiclass tasks using a one-vs-rest strategy.

<p align="center"><img src="https://github.com/user-attachments/assets/c44b7bf7-3126-45db-875e-3fcc85d1a863" alt="SKLearnLogisticRegression" width="200" height="75" />


### SKLearnSVC Component:  
Initializes an Support Vector Classifier (SVC), effective in high-dimensional spaces and suitable for cases with more features than samples.

<p align="center"><img src="https://github.com/user-attachments/assets/c22f156a-9556-40be-93e2-576f6a0f9879" alt="SKLearnSVC" width="200" height="75" />


### SKLearnKNeighborsClassifier Component:  
Initializes a KNeighborsClassifier, an instance-based learning model that classifies data based on stored training instances without building a generalized model.

<p align="center"><img src="https://github.com/user-attachments/assets/b9ba11f9-0679-4bc1-ac66-db89ee620d4a" alt="SKLearnKNeighborsClassifier" width="200" height="75"/>


### SKLearnDecisionTreeClassifier Component:  
Initializes a DecisionTreeClassifier, a versatile model for classification and regression that uses a tree structure to make decisions through yes/no questions.

<p align="center"><img src="https://github.com/user-attachments/assets/781dddf4-1d5e-4794-a3f7-59a961b4eba8" alt="SKLearnDecisionTreeClassifier" width="200" height="75" />


### SKLearnGradientBoostingClassifier Component:  
Initializes a GradientBoostingClassifier that builds models additively in stages, optimizing differentiable loss functions for improved accuracy.

<p align="center"><img src="https://github.com/user-attachments/assets/4db62a79-4f1a-4662-8e51-bea5b1639395" alt="SKLearnGradientBoostingClassifier" width="200" height="75" />


### SKLearnSVR Component:
Initializes a Support Vector Regression (SVR) model, applying Support Vector Machines (SVM) principles to regression with customizable kernels for handling complex datasets.

<p align="center"><img src="https://github.com/user-attachments/assets/4a41da8a-d8a9-4702-8825-9926b7f33e44" alt="SKLearnSVC" width="200" height="75" />


### SKLearnMultinomialNB Component:  
Initializes a MultinomialNB model, ideal for discrete features like word counts and effective for multi-class text classification.

<p align="center"><img src="https://github.com/user-attachments/assets/570706cd-80c1-4563-8731-cea52d814e3a" alt="SKLearnMultinomialNB" width="200" height="75" />


### SKLearnRidgeRegression Component:  
Initializes a Ridge Regression model that mitigates overfitting by penalizing large coefficients, enhancing the robustness of linear regression.

<p align="center"><img src="https://github.com/user-attachments/assets/d44db83d-56d1-4326-8115-8bd0857c1cf2" alt="SKLearnRidgeRegression" width="200" height="75" />


### SKLearnKMeans Component:  
Initializes a KMeans model, an unsupervised algorithm that partitions data into k clusters by assigning each point to the nearest cluster mean.

<p align="center"><img src="https://github.com/user-attachments/assets/76413bd8-e5fd-4fe7-8161-a477497e3545" alt="SKLearnKMeans" width="200" height="75" />

## Try the Examples

We have provided an example workflow to help you get started with the Sklearn component library. Give it a try and see how you can create custom Sklearn components for your applications.

### TrainEvaluate 
Check out the `TrainEvaluate` workflow. This example uses Sklearn components to load the Iris dataset, split it into training and testing sets, and train a Random Forest model. It evaluates the model's performance with classification metrics, showcasing an end-to-end machine learning pipeline.

## Installation

To use this component library, ensure that you have an existing [Xircuits setup](https://xircuits.io/docs/main/Installation). You can then install the SKLearn library using the [component library interface](https://xircuits.io/docs/component-library/installation#installation-using-the-xircuits-library-interface), or through the CLI using:

```
xircuits install sklearn
```

You can also do it manually by cloning and installing it:

```
# base Xircuits directory  
git clone https://github.com/XpressAI/xai-sklearn xai_components/xai_sklearn  
pip install -r xai_components/xai_sklearn/requirements.txt  
```
