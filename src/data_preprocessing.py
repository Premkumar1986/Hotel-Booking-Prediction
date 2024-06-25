import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('data/raw/hotel_bookings.csv')

# Drop 'company' column
data = data.drop(['company'], axis=1)

# Remove duplicates
data_cleaned = data.drop_duplicates()

# Simple imputation
data_cleaned.fillna(0, inplace=True)

# Save processed data
data_cleaned.to_csv('data/processed/hotel_bookings_cleaned.csv', index=False)
