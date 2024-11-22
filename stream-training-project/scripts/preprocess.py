from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

def preprocess_data(df):
    """
    Preprocesses a batch of data: encoding, imputing, and feature extraction.
    """
    # Convert timestamp to datetime and extract features
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month

    # Label encode categorical variables
    label_encoders = {}
    for col in ['Payment Format', 'Receiving Currency', 'Payment Currency']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df[['Amount Received', 'Amount Paid']] = imputer.fit_transform(
        df[['Amount Received', 'Amount Paid']]
    )

    # Drop unused columns
    df = df.drop(['Timestamp', 'Account', 'Account.1'], axis=1)
    return df