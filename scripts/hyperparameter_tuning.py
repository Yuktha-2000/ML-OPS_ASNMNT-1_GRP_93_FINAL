from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
def print_performance_metrics(y_true, y_pred, model):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"{model} Accuracy: {accuracy * 100:.2f}%")
    print(f"{model} Precision: {precision * 100:.2f}%")
    print(f"{model} Recall: {recall * 100:.2f}%")
    print(f"{model} F1-Score: {f1 * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)

def ada_boost_tuning(x_train, y_train, x_test, y_test):
    """Perform hyperparameter tuning on AdaBoost."""
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of boosting stages
        'learning_rate': [0.01, 0.1, 0.5, 1.0],  # Step size for each boosting stage
        'estimator': [DecisionTreeClassifier(max_depth=1),  # Shallow tree as the base estimator
                      None],  # Default is DecisionTreeClassifier
        'algorithm': ['SAMME', 'SAMME.R'],  # Boosting algorithm choice
        'random_state': [42]  # For reproducibility (optional)
    }
    grid_search = GridSearchCV(
        AdaBoostClassifier(random_state=42),
        param_grid,
        scoring='f1_weighted',
        cv=3
    )
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    print(f"Best Parameters: {grid_search.best_params_}")
    print_performance_metrics(y_test, y_pred, 'AdaBoost (Tuned)')