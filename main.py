import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv('amz_uk_processed_data.csv')

# Step 1: Data Cleaning (Remove duplicates and unnecessary columns)
df = df.drop_duplicates(subset=['asin'])  # Remove duplicates
df = df.drop(['imgUrl', 'productURL'], axis=1)  # Drop unnecessary columns

# Step 2: Create 'interaction_score' column
df['interaction_score'] = (df['reviews'] * df['stars']) / df['price']

# Step 3: Handling Missing Values and Infinite Values

# Check for NaN and infinity values
print("Checking for NaN values before preprocessing:")
print(df[['reviews', 'price', 'interaction_score']].isnull().sum())

print("Checking for infinity values before preprocessing:")
print(np.isinf(df[['reviews', 'price', 'interaction_score']]).sum())

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaNs with the median (without inplace=True to avoid FutureWarning)
df['reviews'] = df['reviews'].fillna(df['reviews'].median())
df['price'] = df['price'].fillna(df['price'].median())
df['interaction_score'] = df['interaction_score'].fillna(df['interaction_score'].median())

# Step 4: Check Data Types and Convert to Numeric if Needed
print("Checking data types before scaling:")
print(df[['reviews', 'price', 'interaction_score']].dtypes)

# Convert non-numeric columns (if any) to numeric
df['reviews'] = pd.to_numeric(df['reviews'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['interaction_score'] = pd.to_numeric(df['interaction_score'], errors='coerce')

# Step 5: Scaling the Data
scaler = StandardScaler()

# Apply scaling to 'reviews', 'price', and 'interaction_score'
df[['reviews', 'price', 'interaction_score']] = scaler.fit_transform(df[['reviews', 'price', 'interaction_score']])

# Step 6: Verify the scaling worked and the values are normalized
print("Data after scaling:")
print(df[['reviews', 'price', 'interaction_score']].head())

# Step 7: Save the preprocessed data (optional)
df.to_csv('processed_ecommerce_data.csv', index=False)
print("Preprocessing complete and data saved to 'processed_ecommerce_data.csv'.")


###This code preprocesses an e-commerce dataset to prepare it for use in a machine learning model, specifically for a recommendation system. It cleans the data by removing duplicates and irrelevant columns, creates a useful feature called interaction_score, and handles missing or infinite values to ensure the dataset is complete. The code also ensures that relevant columns are in the correct numeric format and then scales the data so that all features are on a comparable scale, which is important for model performance. Finally, the cleaned and scaled dataset is saved for further use in building a recommendation engine or other predictive models.
