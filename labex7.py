import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Load data from CSV file
def load_data(filename):
    data = pd.read_csv(filename)
    X = data.drop('score', axis=1)  # Assuming the target column is named 'score'
    y = data['score']
    return X, y

# A1. Evaluation of classification performance
def evaluate_classification_performance(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return conf_matrix, accuracy, precision, recall, f1

# A2. Regression performance analysis
def regression_performance_analysis(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_error(y_true, y_pred) / np.mean(y_true) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

# A3. Generating and visualizing training data
def generate_visualize_training_data(X_train, y_train):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[y_train==0][:, 0], X_train[y_train==0][:, 1], c='blue', label='Class 0')
    plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], c='red', label='Class 1')
    plt.title('Training Data')
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.legend()
    plt.show()

# A4. Classification of test data using kNN
def classify_test_data(X_train, y_train, X_test):
    y_pred_knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train).predict(X_test)
    return y_pred_knn

# A7. Hyperparameter tuning using RandomSearchCV
def hyperparameter_tuning(X_train, y_train):
    param_dist = {'n_neighbors': range(1, 11)}
    knn = KNeighborsClassifier()
    random_search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=5, cv=5)
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    return best_params

# Main function
def main():
    # Load data
    filename = r'C:\Users\Murari\Downloads\code_comm.csv'  # Replace with the actual file path
    X, y = load_data(filename)
    
    # For classification tasks
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # A1. Evaluation of classification performance
    # Example usage:
    y_pred = classify_test_data(X_train, y_train, X_test)
    conf_matrix, accuracy, precision, recall, f1 = evaluate_classification_performance(y_test, y_pred)
    
    print("Confusion Matrix:\n", conf_matrix)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    
    # A2. Regression performance analysis
    # Example usage:
    # y_pred_reg = your_regression_model.predict(X_test)
    # mse, rmse, mape, r2 = regression_performance_analysis(y_test, y_pred_reg)
    
    # A3. Generating and visualizing training data
    # Example usage:
    # generate_visualize_training_data(X_train, y_train)
    
    # A7. Hyperparameter tuning using RandomSearchCV
    # best_params = hyperparameter_tuning(X_train, y_train)
    
if _name_ == "_main_":
    main()
