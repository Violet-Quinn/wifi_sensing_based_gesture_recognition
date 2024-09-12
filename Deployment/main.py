from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import pandas as pd
import re
from math import sqrt, atan2
import joblib


def process_csv(filename):
    d = pd.read_csv(filename)

    amps = []
    phases = []

    for i, j in enumerate(d['data']):
        imaginary = []
        real = []
        amp = []
        ph = []

        csi_string = re.findall(r"\[(.*)]", j)[0]
        csi_raw = [int(x) for x in csi_string.split(",") if x != '']

        for k in range(0, len(csi_raw), 2):
            imaginary.append(csi_raw[k])
            real.append(csi_raw[k + 1])

        for k in range(len(imaginary)):
            amp.append(round(sqrt(imaginary[k] ** 2 + real[k] ** 2), 1))
            ph.append(round(atan2(imaginary[k], real[k])))

        amps.append(amp)
        phases.append(ph)

    # Combine amps and phases into a single list
    result_list = [item for sublist in (amps + phases) for item in sublist]

    return result_list


def combine_results(file_list):
    result_matrix = []
    labels = []

    for filename in file_list:
        result_list = process_csv(filename)
        result_matrix.append(result_list)

        # Extract label from the file name
        if "bye" in filename:
            label = "bye"
        elif "hello" in filename:
            label = "hello"
        elif "thankyou" in filename:
            label = "thankyou"
        elif "yes" in filename:
            label = "yes"
        else:
            label = "unknown"  # You can handle other cases as needed

        labels.append(label)

    # Create a DataFrame from the result matrix
    df = pd.DataFrame(result_matrix)

    # Add a column with the labels
    df['label'] = labels

    return df


# Example usage with multiple files:
file_list = [
    r"D:\Wifi Sensing Project\Deployment\data\bye_s1_1.csv",
    r"D:\Wifi Sensing Project\Deployment\data\bye_s1_2.csv",
    r"D:\Wifi Sensing Project\Deployment\data\bye_s1_3.csv",
    r"D:\Wifi Sensing Project\Deployment\data\bye_s1_4.csv",
    r"D:\Wifi Sensing Project\Deployment\data\bye_s1_5.csv",
    r"D:\Wifi Sensing Project\Deployment\data\bye_s2_1.csv",
    r"D:\Wifi Sensing Project\Deployment\data\bye_s2_2.csv",
    r"D:\Wifi Sensing Project\Deployment\data\bye_s2_3.csv",
    r"D:\Wifi Sensing Project\Deployment\data\bye_s2_4.csv",
    r"D:\Wifi Sensing Project\Deployment\data\bye_s2_5.csv",

    r"D:\Wifi Sensing Project\Deployment\data\hello_s1_1.csv",
    r"D:\Wifi Sensing Project\Deployment\data\hello_s1_2.csv",
    r"D:\Wifi Sensing Project\Deployment\data\hello_s1_3.csv",
    r"D:\Wifi Sensing Project\Deployment\data\hello_s1_4.csv",
    r"D:\Wifi Sensing Project\Deployment\data\hello_s1_5.csv",
    r"D:\Wifi Sensing Project\Deployment\data\hello_s2_1.csv",
    r"D:\Wifi Sensing Project\Deployment\data\hello_s2_2.csv",
    r"D:\Wifi Sensing Project\Deployment\data\hello_s2_3.csv",
    r"D:\Wifi Sensing Project\Deployment\data\hello_s2_4.csv",
    r"D:\Wifi Sensing Project\Deployment\data\hello_s2_5.csv",

    r"D:\Wifi Sensing Project\Deployment\data\thankyou_s1_1.csv",
    r"D:\Wifi Sensing Project\Deployment\data\thankyou_s1_2.csv",
    r"D:\Wifi Sensing Project\Deployment\data\thankyou_s1_3.csv",
    r"D:\Wifi Sensing Project\Deployment\data\thankyou_s1_4.csv",
    r"D:\Wifi Sensing Project\Deployment\data\thankyou_s1_5.csv",
    r"D:\Wifi Sensing Project\Deployment\data\thankyou_s2_1.csv",
    r"D:\Wifi Sensing Project\Deployment\data\thankyou_s2_2.csv",
    r"D:\Wifi Sensing Project\Deployment\data\thankyou_s2_3.csv",
    r"D:\Wifi Sensing Project\Deployment\data\thankyou_s2_4.csv",
    r"D:\Wifi Sensing Project\Deployment\data\thankyou_s2_5.csv",

    r"D:\Wifi Sensing Project\Deployment\data\yes_s1_1.csv",
    r"D:\Wifi Sensing Project\Deployment\data\yes_s1_2.csv",
    r"D:\Wifi Sensing Project\Deployment\data\yes_s1_3.csv",
    r"D:\Wifi Sensing Project\Deployment\data\yes_s1_4.csv",
    r"D:\Wifi Sensing Project\Deployment\data\yes_s1_5.csv",
    r"D:\Wifi Sensing Project\Deployment\data\yes_s2_1.csv",
    r"D:\Wifi Sensing Project\Deployment\data\yes_s2_2.csv",
    r"D:\Wifi Sensing Project\Deployment\data\yes_s2_3.csv",
    r"D:\Wifi Sensing Project\Deployment\data\yes_s2_4.csv",
    r"D:\Wifi Sensing Project\Deployment\data\yes_s2_5.csv"
]

result_df = combine_results(file_list)

# Display the result DataFrame
print("Number of input files:", len(file_list))
print("Number of successfully classified files:")

# Extract features (X) and labels (y)
X = result_df.iloc[:, :-1].values
y = result_df['label'].values

# Impute missing values with the mean (you can choose a different strategy based on your data)
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Split the data into training and testing sets after imputation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Define the hyperparameter search space
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# Create a RandomForestClassifier
classifier = RandomForestClassifier(random_state=42)

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    classifier,
    param_distributions=param_dist,
    n_iter=10,  # Number of random combinations to try
    cv=5,  # Number of cross-validation folds
    random_state=42
)

# Fit the model
random_search.fit(X_train, y_train)

# Get the best parameters and retrain the model
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the classifier with the best hyperparameters
classifier = RandomForestClassifier(**best_params, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
result_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result_matrix)

result_classification_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(result_classification_report)

result_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", result_accuracy)
print("Number of successfully classified files:", len(y_test))

# Train the classifier with the best hyperparameters
classifier = RandomForestClassifier(**best_params, random_state=42)
classifier.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'trained_model.joblib'
joblib.dump(classifier, model_filename)

# Save the imputer used for preprocessing
imputer_filename = 'imputer.joblib'
joblib.dump(imputer, imputer_filename)
