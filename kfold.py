import random
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

class CustomClassifier:
    def __init__(self, classifier, param_choices, n_splits=2):
        self.classifier = classifier
        self.param_choices = param_choices
        self.f1_score = 0
        self.n_splits = n_splits

    def create_random(self):
        params = {}
        for key in self.param_choices:
            params[key] = random.choice(self.param_choices[key])
        self.classifier.set_params(**params)

    def train(self, x, y):
        kf = KFold(n_splits=self.n_splits, shuffle=True)
        f1_scores = []
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.classifier.fit(x_train, y_train)
            y_pred = self.classifier.predict(x_test)
            f1_scores.append(f1_score(y_test, y_pred))

        self.f1_score = sum(f1_scores) / len(f1_scores)  # Average F1 score across folds

class Optimizer:
    def __init__(self, param_choices, retain=0.4, mutate_chance=0.2):
        self.mutate_chance = mutate_chance
        self.retain = retain
        self.param_choices = param_choices

    def create_population(self, count, classifier, x, y):
        population = []
        for _ in tqdm(range(count), desc="Creating Population"):
            custom_classifier = CustomClassifier(classifier, self.param_choices)
            custom_classifier.create_random()
            custom_classifier.train(x, y)
            population.append(custom_classifier)
        return population

    def fitness(self, classifier):
        return classifier.f1_score

    def breed(self, mother, father, x, y):
        children = []
        child1, child2 = {}, {}
        for param in self.param_choices:
            if random.choice([0, 1]) == 0:
                child1[param] = father.classifier.get_params()[param]
                child2[param] = mother.classifier.get_params()[param]
            else:
                child1[param] = mother.classifier.get_params()[param]
                child2[param] = father.classifier.get_params()[param]

        classifier1 = CustomClassifier(mother.classifier.__class__(**child1), self.param_choices)
        classifier2 = CustomClassifier(mother.classifier.__class__(**child2), self.param_choices)
        if self.mutate_chance > random.random():
            classifier1 = self.mutate(classifier1)
        if self.mutate_chance > random.random():
            classifier2 = self.mutate(classifier2)

        classifier1.train(x, y)
        classifier2.train(x, y)

        children.append(classifier1)
        children.append(classifier2)

        return children

    def mutate(self, classifier):
        mutation = random.choice(list(self.param_choices.keys()))
        classifier.classifier.set_params(**{mutation: random.choice(self.param_choices[mutation])})
        return classifier

    def evolve(self, pop, x, y):
        graded = [(self.fitness(classifier), classifier) for classifier in pop]
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
        retain_length = int(len(graded) * self.retain)
        parents = graded[:retain_length]

        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []
        while len(children) < desired_length:
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)
            if male != female:
                male = parents[male]
                female = parents[female]
                babies = self.breed(male, female, x, y)
                for baby in babies:
                    if len(children) < desired_length:
                        children.append(baby)
        parents.extend(children)
        return parents

class EvolutionaryProcess:
    def __init__(self, generations, population, params, classifier):
        self._generations = generations
        self._population = population
        self._params = params
        self.classifiers = None
        self.best_params = None
        self.classifier = classifier

    def evolve(self, x, y):
        optimizer = Optimizer(self._params)
        self.classifiers = optimizer.create_population(self._population, self.classifier, x, y)

        for _ in tqdm(range(self._generations), desc="Evolution Progress"):
            self._train_classifiers(x, y)
            self.classifiers = optimizer.evolve(self.classifiers, x, y)

        self.classifiers = sorted(self.classifiers, key=lambda x: x.f1_score, reverse=True)
        self.best_params = self.classifiers[0]
        print("Best F1 score: {}, Best params: {}".format(self.best_params.f1_score, self.best_params.classifier.get_params()))

    def _train_classifiers(self, x, y):
        for classifier in self.classifiers:
            classifier.train(x, y)

heart_data = pd.read_csv("heart_cleveland_upload.csv")

heart_data["thalach"] = pd.cut(heart_data["thalach"], 8, labels=range(1, 9))
heart_data["trestbps"] = pd.cut(heart_data["trestbps"], 5, labels=range(8, 13))
heart_data["age"] = pd.cut(heart_data["age"], 12, labels=range(12, 24))
heart_data["chol"] = pd.cut(heart_data["chol"], 10, labels=range(24, 34))
heart_data["oldpeak"] = pd.cut(heart_data["oldpeak"], 5, labels=range(34, 39))

a = pd.get_dummies(heart_data, columns=["cp", "restecg", "slope", "thalach", "trestbps", "age", "chol", "thal", "oldpeak"], 
                   prefix=["cp", "restecg", "slope", "thalach", "trestbps", "age", "chol", "thal", "oldpeak"], drop_first=True)

x = heart_data.drop(columns=["condition"]).values
y = heart_data["condition"].values

generations = 5
population = 20

param_choices = {
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 120, 230, 340, 450, 560, 670, 780, 890, 1000],
    'min_samples_split': [2, 5, 10, 14],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'criterion': ['entropy', 'gini']
}
classifier = RandomForestClassifier()

# param_choices = {
#     'n_neighbors': [3, 5, 7, 9, 11],
#     'weights': ['uniform', 'distance'],
#     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#     'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
#     'p': [1, 2]
# }
# classifier = KNeighborsClassifier()


# param_choices = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [3, 5, 7],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'subsample': [0.8, 0.9, 1.0],
#     'max_features': ['sqrt', 'log2', None],
#     'loss': ['log_loss', 'exponential'],
#     'criterion': ['friedman_mse', 'squared_error']
# }

# classifier = GradientBoostingClassifier()

evolution = EvolutionaryProcess(generations, population, param_choices, classifier)
evolution.evolve(x, y)
print("Best parameters:", evolution.best_params.classifier.get_params())
