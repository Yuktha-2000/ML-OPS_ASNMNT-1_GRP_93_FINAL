import sys
from scripts.preprocessing import preprocess_data, load_data, get_column_definitions, split_data,save_best_model
from scripts.hyperparameter_tuning import ada_boost_tuning
import joblib
from scripts.deployment.app import run_app

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        # Run the Flask app
        with open('models/best_model.joblib', 'rb') as f:
           model = joblib.load(f)

        print("model loaded succesfully")
        run_app()
    else:
        # Load and preprocess data
        df = load_data("data/Transactions_Dataset.csv")
        cat_cols, num_cols, scale_cols, ordinal_cols, reputation_order, transaction_freq_order = get_column_definitions()
        X, y = preprocess_data(df, cat_cols, scale_cols, ordinal_cols, reputation_order, transaction_freq_order)

        # Train/test split
        x_train, x_test, y_train, y_test = split_data(X, y)

        #Perform hyperparameter tuning
        best_model = ada_boost_tuning(x_train, y_train, x_test, y_test)

        # Save the best model
        save_best_model(best_model, "models/best_model.joblib")
