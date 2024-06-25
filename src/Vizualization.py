import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data with features
data = pd.read_csv('data/processed/hotel_bookings_features.csv')

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('results/correlation_heatmap.png')
plt.show()

# Find the outlier using box plot
def bar_plot(df):
    plt.figure(figsize=(10,10))
    for i, col in enumerate(final.columns):
        plt.subplot(len(final.columns)//2 + len(final.columns)%2, 2,i+1)
        sns.boxplot(y=final[col])
        plt.title(f'Boxplot of {col}')
        
    plt.tight_layout()
    plt.show()

bar_plot(final)

# Find the Price of Room per night per person
new_data = df[df['is_canceled']==0]
plt.figure(figsize=(12,6))
sns.boxplot(x ='reserved_room_type',
            y = 'adr', 
            hue ='hotel', data = new_data)
plt.title('Price per room per person for a night', fontsize = 20)
plt.xlabel('Room Type')
plt.ylabel('Price of Room[in Ero]')
plt.legend(loc ='upper right')
plt.ylim(0,550)
plt.show()

# Visualize the hotel booking countrywise.
import plotly.express as px
map_guest = px.choropleth(country_wise_customer, locations =country_wise_customer['Country Name'],
                         color = country_wise_customer['No.of Guests'],
                         hover_name = country_wise_customer['Country Name'],
                         title = 'Home country of guests')
map_guest.show()
