



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

To use this component library, ensure that you have an existing [Xircuits setup](https://xircuits.io/docs/main/Installation). You can then install the SKLearn library using the [component library interface](https://xircuits.io/docs/component-library/installation#installation-using-the-xircuits-library-interface), or through the CLI using:

```
xircuits install sklearn
```

You can also do it manually by cloning and installing it.

```bash
# To clone the repository into your Xircuits project directory
git clone https://github.com/XpressAI/xai-sklearn.git xai_components/xai_sklearn

# Install required dependencies
pip install -r xai_components/xai_sklearn/requirements.txt
```

## Getting Started with SKLearn XAI Components



Now that you have installed the required libraries and components, you can begin using the SKLearn XAI Components Library to build machine learning workflows in Xircuits. Please follow the documentation and examples provided in the library to learn how to create, customize, and manage machine learning components using SKLearn XAI.


## Try the Example

We have provided an example workflow to help you get started with the SKLearn XAI Components Library. Give it a try and see how you can create a custom machine learning workflow for your projects.

### Train Evaluate

This example demonstrates a machine learning workflow in Xircuits using the SKLearn XAI Components Library. It creates a pipeline for loading the Iris dataset, splitting it, training a RandomForestClassifier, and evaluating its performance


## Components Library

The SKLearn XAI Components Library offers a variety of components designed to facilitate every stage of the machine learning process. You are encouraged to explore these components and consult their documentation to enhance your understanding and application in building effective machine learning workflows.


## Contributing

We welcome contributions to the **SKLearn XAI Components Library**! If you would like to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Open a pull request with a detailed description of your changes.

Please feel free to suggest new components, improvements, or optimizations. If you encounter any issues or have ideas for enhancements, you can open an issue in the repository.

---

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

