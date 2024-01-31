from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, xai_component

@xai_component
class SKLearnRandomForestClassifier(Component):
    """
    Initializes a RandomForestClassifier model with given parameters. RandomForestClassifier is suitable for a wide range of classification tasks and is known for its high accuracy.

    #### Reference:
    - [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

    ##### inPorts:
    - model_params: A dictionary of parameters to initialize the model with. If not provided, default parameters will be used.

    ##### outPorts:
    - model: The initialized RandomForestClassifier model, ready for training with datasets.
    """
    model_params: InArg[dict]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        from sklearn.ensemble import RandomForestClassifier
        params = self.model_params.value or {}
        print("Initializing RandomForestClassifier with parameters:", params)
        self.model.value = RandomForestClassifier(**params)
        print("RandomForestClassifier initialized successfully.")

@xai_component
class SKLearnLogisticRegression(Component):
    """
    Initializes a LogisticRegression model with given parameters. LogisticRegression is a popular model for binary classification problems, as well as multiclass problems in a one-vs-rest scheme.

    #### Reference:
    - [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

    ##### inPorts:
    - model_params: A dictionary of parameters to initialize the model with. If not provided, default parameters will be used.

    ##### outPorts:
    - model: The initialized LogisticRegression model, ready for training with datasets.
    """
    model_params: InArg[dict]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        from sklearn.linear_model import LogisticRegression
        params = self.model_params.value or {}
        print("Initializing LogisticRegression with parameters:", params)
        self.model.value = LogisticRegression(**params)
        print("LogisticRegression initialized successfully.")

@xai_component
class SKLearnSVC(Component):
    """
    Initializes a Support Vector Classifier (SVC) model with given parameters. SVC is effective in high dimensional spaces and for cases where the number of dimensions exceeds the number of samples.

    #### Reference:
    - [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

    ##### inPorts:
    - model_params: A dictionary of parameters to initialize the model with. If not provided, default parameters will be used.

    ##### outPorts:
    - model: The initialized SVC model, ready for training with datasets.
    """
    model_params: InArg[dict]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        from sklearn.svm import SVC
        params = self.model_params.value or {}
        print("Initializing SVC with parameters:", params)
        self.model.value = SVC(**params)
        print("SVC initialized successfully.")

@xai_component
class SKLearnKNeighborsClassifier(Component):
    """
    Initializes a KNeighborsClassifier model with given parameters. KNeighborsClassifier is a type of instance-based learning or non-generalizing learning that does not attempt to construct a general internal model but stores instances of the training data.

    #### Reference:
    - [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

    ##### inPorts:
    - model_params: A dictionary of parameters to initialize the model with. If not provided, default parameters will be used.

    ##### outPorts:
    - model: The initialized KNeighborsClassifier model, ready for training with datasets.
    """
    model_params: InArg[dict]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        from sklearn.neighbors import KNeighborsClassifier
        params = self.model_params.value or {}
        print("Initializing KNeighborsClassifier with parameters:", params)
        self.model.value = KNeighborsClassifier(**params)
        print("KNeighborsClassifier initialized successfully.")

@xai_component
class SKLearnDecisionTreeClassifier(Component):
    """
    Initializes a DecisionTreeClassifier model with given parameters. DecisionTreeClassifier is a powerful model for classification and regression tasks. It constructs a tree for decision-making with yes/no questions.

    #### Reference:
    - [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

    ##### inPorts:
    - model_params: A dictionary of parameters to initialize the model with. If not provided, default parameters will be used.

    ##### outPorts:
    - model: The initialized DecisionTreeClassifier model, ready for training with datasets.
    """
    model_params: InArg[dict]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        from sklearn.tree import DecisionTreeClassifier
        params = self.model_params.value or {}
        print("Initializing DecisionTreeClassifier with parameters:", params)
        self.model.value = DecisionTreeClassifier(**params)
        print("DecisionTreeClassifier initialized successfully.")

@xai_component
class SKLearnGradientBoostingClassifier(Component):
    """
    Initializes a GradientBoostingClassifier model with given parameters. GradientBoostingClassifier builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.

    #### Reference:
    - [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

    ##### inPorts:
    - model_params: A dictionary of parameters to initialize the model with. If not provided, default parameters will be used.

    ##### outPorts:
    - model: The initialized GradientBoostingClassifier model, ready for training with datasets.
    """
    model_params: InArg[dict]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        from sklearn.ensemble import GradientBoostingClassifier
        params = self.model_params.value or {}
        print("Initializing GradientBoostingClassifier with parameters:", params)
        self.model.value = GradientBoostingClassifier(**params)
        print("GradientBoostingClassifier initialized successfully.")

@xai_component
class SKLearnSVR(Component):
    """
    Initializes a Support Vector Regression (SVR) model with given parameters. SVR applies the principles of Support Vector Machines (SVM) for regression tasks, providing a flexible choice of kernels and the possibility of tuning for complex datasets.

    #### Reference:
    - [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

    ##### inPorts:
    - model_params: A dictionary of parameters to initialize the model with. If not provided, default parameters will be used.

    ##### outPorts:
    - model: The initialized SVR model, ready for training with datasets.
    """
    model_params: InArg[dict]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        from sklearn.svm import SVR
        params = self.model_params.value or {}
        print("Initializing SVR with parameters:", params)
        self.model.value = SVR(**params)
        print("SVR initialized successfully.")

@xai_component
class SKLearnMultinomialNB(Component):
    """
    Initializes a Multinomial Naive Bayes (MultinomialNB) model with given parameters. MultinomialNB is particularly suited for discrete features (e.g., text classification with word counts) and can handle multiple classes.

    #### Reference:
    - [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)

    ##### inPorts:
    - model_params: A dictionary of parameters to initialize the model with. If not provided, default parameters will be used.

    ##### outPorts:
    - model: The initialized MultinomialNB model, ready for training with datasets.
    """
    model_params: InArg[dict]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        from sklearn.naive_bayes import MultinomialNB
        params = self.model_params.value or {}
        print("Initializing MultinomialNB with parameters:", params)
        self.model.value = MultinomialNB(**params)
        print("MultinomialNB initialized successfully.")

@xai_component
class SKLearnRidgeRegression(Component):
    """
    Initializes a Ridge Regression model with given parameters. Ridge Regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of coefficients to prevent overfitting.

    #### Reference:
    - [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

    ##### inPorts:
    - model_params: A dictionary of parameters to initialize the model with. If not provided, default parameters will be used.

    ##### outPorts:
    - model: The initialized Ridge Regression model, ready for training with datasets.
    """
    model_params: InArg[dict]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        from sklearn.linear_model import Ridge
        params = self.model_params.value or {}
        print("Initializing Ridge Regression with parameters:", params)
        self.model.value = Ridge(**params)
        print("Ridge Regression initialized successfully.")

@xai_component
class SKLearnKMeans(Component):
    """
    Initializes a KMeans clustering model with given parameters. KMeans is a popular unsupervised learning algorithm for cluster analysis in data mining. KMeans clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean.

    #### Reference:
    - [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

    ##### inPorts:
    - model_params: A dictionary of parameters to initialize the model with. If not provided, default parameters will be used.

    ##### outPorts:
    - model: The initialized KMeans model, ready for clustering tasks.
    """
    model_params: InArg[dict]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        from sklearn.cluster import KMeans
        params = self.model_params.value or {}
        print("Initializing KMeans with parameters:", params)
        self.model.value = KMeans(**params)
        print("KMeans initialized successfully.")
