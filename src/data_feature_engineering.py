import pandas as pd

# Load cleaned data
data = pd.read_csv('data/processed/hotel_bookings_cleaned.csv')

# Feature engineering
data['stay'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'])
data['year'] = data['reservation_status_date'].dt.year
data['month'] = data['reservation_status_date'].dt.month
data['day'] = data['reservation_status_date'].dt.day

# Save features
data.to_csv('data/processed/hotel_bookings_features.csv', index=False)
