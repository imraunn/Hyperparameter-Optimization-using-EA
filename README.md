# Hyperparameter Optimization using Evolutionary Algorithm

This repository contains Python code for hyperparameter optimization using an evolutionary algorithm. The algorithm aims to find optimal hyperparameters for a given machine learning classifier using the F1 score as the metric. This code is inspired by Alexander Osipenko's repository: https://github.com/subpath/neuro-evolution/

## Usage

1. Clone the repository:

```bash
git clone https://github.com/imraunn/hyperparameter-optimization.git
```

2. Install the required dependencies:

```bash
pip install pandas scikit-learn tqdm
```

3. Run the hyperparameter optimization script with train_test_split

```bash
python main.py
```

Or, run the hyperparameter optimization script with KFold Cross Validation:
```bash
python kfold.py
```
## Code Overview

The code consists of the following main components:

#### 1. CustomClassifier

- This class defines a custom classifier with methods to create random hyperparameters and train the classifier using k-fold cross-validation.

#### 2. Optimizer

- This class implements methods for creating a population of custom classifiers, calculating fitness, breeding, mutating, and evolving the population.

#### 3. EvolutionaryProcess

- This class orchestrates the evolution process by creating an initial population of custom classifiers, evolving them over generations, and identifying the best set of hyperparameters.

#### 4. Hyperparameter Optimization Script

- The script reads data from a CSV file, preprocesses it, defines the parameter choices for the classifier (Random Forest, Gradient Boosting, or K-Nearest Neighbors), and executes the evolutionary process to find the best hyperparameters.

## Example

An example of hyperparameter optimization using a Random Forest classifier is provided in the script `main.py`. You can uncomment and modify the parameter choices and classifier type according to your requirements.

## Dataset

The code uses the Heart Disease dataset (`heart_cleveland_upload.csv`) for demonstration purposes. You can replace it with your own dataset by modifying the data loading part of the script.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- tqdm
