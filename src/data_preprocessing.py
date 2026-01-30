import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Drop unnecessary columns
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Geography'] = le.fit_transform(df['Geography'])

    # Split features and target
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.columns