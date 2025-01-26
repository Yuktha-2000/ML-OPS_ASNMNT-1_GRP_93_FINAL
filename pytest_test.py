from sklearn.metrics import accuracy_score
import pytest
import joblib
from sklearn.model_selection import train_test_split
from scripts.preprocessing import preprocess_data, load_data, get_column_definitions, split_data

def t_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


# Fixture to load and split the data
@pytest.fixture
def transaction_data():
    df = load_data("data/Transactions_Dataset.csv")
    cat_cols, num_cols, scale_cols, ordinal_cols, reputation_order, transaction_freq_order = get_column_definitions()
    X, y = preprocess_data(df, cat_cols, scale_cols, ordinal_cols, reputation_order, transaction_freq_order)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test

# Fixture to load the pre-trained model
@pytest.fixture
def pre_trained_model():
    # Load the pre-trained model using joblib
    model = joblib.load('models/best_model.joblib')  # Assuming the model is saved as 'model.pkl'
    return model

# Test the model's accuracy using the pre-trained model
def test_model_accuracy(transaction_data, pre_trained_model):
    _, X_test, _, y_test = transaction_data  # Unpack the data
    accuracy = t_model(X_test, y_test, pre_trained_model)  # Test the pre-trained model's accuracy
    assert accuracy > 0.9, f"Model accuracy should be greater than 90%, but got {accuracy:.2f}"