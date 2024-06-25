#!/usr/bin/env python
# coding: utf-8

# ***MY FIRST PROJECT***

# Import the necessary libraries for this projects

# In[1]:


import pandas as pd # for Data Manipulation & Analysis
import numpy as np # for numerical & Matrix calculations
import matplotlib.pyplot as plt # for visualization
import seaborn as sns # for higher level of visualization
import warnings # to avoid warning messages
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ##### Import the dataset 

# In[2]:


data = pd.read_csv('E:/My Learning/My Projects/Solar Secure/Project 1/Data/hotel_bookings.csv') # Read the dataset and show it
data.head(5)


# In[3]:


data.info()


# From this we found that there are 4- float datatype, 16 int datatype & 12- object datatype columns.

# **Find how many rows and columns are there in this data set?**

# In[4]:


data.shape


# There are 119390 rows and 32 columns in this dataset.

# **Statistical report about the Hotel Booking Dataset**

# In[5]:


data.describe(include='number').T


# In[6]:


data.describe(include ='object').T


# **Find the null values in the Dataset**

# In[7]:


data.isnull().sum()


# out these feature 'company' is missing completely at Random(MCAR), 'agent' is missing at random(MAR) & for 'country' & 'children' missing not at random (MNAR)

# **Data Cleaning**

# As we have seen that 'Company' has MCAR. so we simply drop this column, instead of imputation.

# In[8]:


data = data.drop(['company'], axis =1)
data.shape


# In[9]:


# Plot the missing values 
import klib
klib.missingval_plot(data)


# **Find the duplicate values in the data set**

# In[10]:


data.duplicated().sum()


# In[11]:


if data.duplicated().any():
    print("There are duplicate rows in the DataFrame.")
    
    # View the duplicated rows
    duplicated_rows = data[data.duplicated(keep=False)]
    print("Duplicated Rows:")
    print(duplicated_rows)


# Duplicate data due to same customer enquired about the booking for many times but not really booked. So no need to keep this duplicate values for model building.

# In[12]:


# Remove the duplicated rows
data_cleaned = data.drop_duplicates()
print("DataFrame after removing duplicates:")
print(data_cleaned.shape)


# **imputation of missing values for Children, agent & country variables.**

# In[116]:


data1 = data_cleaned.copy() # copy the dataset


# In[14]:


def simple_impute(df):
    df.fillna(0, inplace =True) # simple zero imputaion
    print(df.isnull().sum())


# In[15]:


simple_impute(data1) # After imputation zero null values


# Now there is no null value in our data set.

# **Find the correlation for numeric variable** 

# In[16]:


num_col = data1.select_dtypes(include ='number').columns
df_corr =data[num_col].corr()
is_canceled_corr = df_corr['is_canceled']
print(is_canceled_corr)


# **Visualize the correlation using clustermap**

# In[17]:


sns.clustermap(df_corr, cmap='coolwarm', annot=True)
plt.show()


# Removing or dropping a feature based on correlation in a heatmap can help simplify your model and potentially improve its performance by eliminating redundant or highly correlated features. 

# Find the highly correlated variables

# In[18]:


plt.figure(1,figsize =(10,10))
sns.heatmap(df_corr, annot =True, linewidths =0.5, linecolor ='red')


# In[19]:


corr_matrix =df_corr 
threshold = 0.5 # Pearson Correlation value


# In[20]:


# Find index pairs of highly correlated features
high_corr_pairs = [(corr_matrix.columns[i], corr_matrix.columns[j]) 
                   for i in range(len(corr_matrix.columns)) 
                   for j in range(i) 
                   if abs(corr_matrix.iloc[i, j]) > threshold]

print("Highly correlated pairs:")
for pair in high_corr_pairs:
    print(pair)


# **We have to find the important features for this problem**

# Goal: Maximize the hotel booking confirmation. Inorder to achive this goal, we have to reduce the no.of cancellations, Automate confirmations, sell extra facilites, etc.
#     Extra facilites/services - like free vehicle parking, child care, free cancellation, free meals/breakfast, concession for off seasons, value customer services, etc.

# Here both children and babies can't stay in hotels without their parents. So we have check the details about it.

# In[21]:


list1 = ['children', 'adults','babies']
for i in list1:
    print(f'unique values of {i} is',data1[i].unique())


# In[22]:


filter_data = (data1['children']==0)&(data1['adults']==0)&(data1['babies']==0) 
# remove the row whose values are zero for all children, babies & children. which means they have booked, but not visited the hotel
filter_data = data1[~filter_data] # remove the particular rows form the data set.


# In[23]:


filter_data.shape


# **Find country wise customer visited our hotel ?**

# In[24]:


df = filter_data
df['country'].unique()


# In[25]:


(df['is_canceled']==0).value_counts() # True value indicates customer visited our hotel whereas false indicate only enquiry about booking


# In[26]:


country_wise_customer =df[df['is_canceled']==0]['country'].value_counts().reset_index()


# In[27]:


country_wise_customer.columns = ['Country Name', 'No.of Guests'] 
country_wise_customer # rename the column names in the dataset


# **visualize the hotel booking using plotly library**

# In[28]:


import plotly.express as px
map_guest = px.choropleth(country_wise_customer, locations =country_wise_customer['Country Name'],
                         color = country_wise_customer['No.of Guests'],
                         hover_name = country_wise_customer['Country Name'],
                         title = 'Home country of guests')
map_guest.show()


# **How much a guest paid for a room per day?**

# In[29]:


# First find the types of hotels available?
df['hotel'].unique()


# In[30]:


sns.catplot(x= 'hotel', hue = 'is_canceled', kind= 'count', data = df) 


# `Visualization of customer/guests booked hotels` 

# *Find the cost of hotel?*

# In[31]:


df['adr']


# *cost of hotel varies according to their room size, no. of cot, meals and other facilities* 

# In[32]:


new_data = df[df['is_canceled']==0] # we choose only visted guests


# In[33]:


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


# **Hoe does the price per night vary over the year?**

# In[34]:


resort_data = df[(df['hotel']== 'Resort Hotel') & (df['is_canceled']==0)]
city_data = df[(df['hotel'] == 'City Hotel') & (df['is_canceled']==0)]
resort_data.head(5)


# In[35]:


city_data.head(5)


# In[36]:


price = resort_data.groupby(['arrival_date_month'])['adr'].mean() # Average price of the room in resort hotel per person as monthwise
price = price.reset_index()
price.columns = ['Arrival Month', 'Price']
resort_hotel = price.sort_values(by ='Price')
resort_hotel


# Highest price of rooms are during Jun, July, Aug & Sep month. This is may be due to summer vacation

# In[37]:


city_hotel = city_data.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel.columns = ['Arrival Month', 'Price']
city_hotel


# In[38]:


final = resort_hotel.merge(city_hotel, on ='Arrival Month')
final.columns = ['Arrival Months', 'Price_Resort','Price_city']
final


# In[39]:


final.columns


# In[40]:


final.sort_values(by ='Arrival Months')


# **Arranging the price as per Month wise by using mapping method**

# In[41]:


print(type(final))


# In[42]:


# Mapping of of months to numerical values
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
# Add a new column for numerical month values
final['Months'] = final['Arrival Months'].map(month_map)
sorted_final = final.sort_values(by ='Arrival Months')

sorted_final = sorted_final.drop(columns =['Months'])
print(sorted_final)


# **Monthwise pricing room rent by using categorical method**

# In[43]:


# Define the order of the months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Convert 'Arrival Months' to a categorical type with the specified order
final['Arrival Months'] = pd.Categorical(final['Arrival Months'], categories=month_order, ordered=True)

# Sort the DataFrame by 'Arrival Months'
df_sorted = final.sort_values(by='Arrival Months')

print(df_sorted)


# In[44]:


# Arrange Monthwise by using loop function
from calendar import month_name

def sort_month(df, colname):
    month_dict = {j : i for i ,j in enumerate(month_name)} # dictionary comprehension
    df['month_num'] = df[colname].apply(lambda x:month_dict[x])
    return df.sort_values(by ='month_num').reset_index().drop(['month_num','index'], axis =1)


# In[45]:


sort_month(final, 'Arrival Months')


# In[46]:


sorted_final.plot(kind='line', x ='Arrival Months', y = ['Price_Resort','Price_city'])


# ***Find the number of Guests visted our hotels in monthwise***

# In[47]:


num_guest_resort = resort_data['arrival_date_month'].value_counts().reset_index()
num_guest_resort.columns = ['Months','No.of Guests']
num_guest_resort


# In[48]:


num_guest_city = city_data['arrival_date_month'].value_counts().reset_index()
num_guest_city.columns = ['Months','No.of Guests']
num_guest_city


# In[49]:


num_guest_final = num_guest_resort.merge(num_guest_city, on = 'Months')
num_guest_final.columns = ['Months','Resort Guest No', 'City Guest No']
num_guest_final


# In[50]:


sorted_month = sort_month(num_guest_final, 'Months')
sorted_month


# In[51]:


sorted_month.plot(kind='line', x='Months', y =['Resort Guest No','City Guest No'])


# ***For the above graph we find that the number of guests visited city_hotel is higher throughout the year. This is because of low price, rooms avalibility, etc reasons.***

# **Find the number of days a guest stayed in our hotels?**

# In[52]:


final_data = (df['is_canceled'] ==0) # get only visited customer data
clean_data = df[final_data] # Store only the visited customer data
clean_data.head(5)


# In[53]:


clean_data['stay'] = clean_data['stays_in_weekend_nights'] + clean_data['stays_in_week_nights']
clean_data.head(5)


# In[54]:


stay = clean_data.groupby(['stay','hotel']).agg('count').reset_index()
stay = stay.iloc[:,0:3]
stay


# In[55]:


stay = stay.rename(columns ={'is_canceled': 'No of Days Stay'})
stay


# In[56]:


sns.barplot(x ='stay',y ='No of Days Stay', hue='hotel',hue_order =['City Hotel','Resort Hotel'], data= stay)
plt.legend(loc = 'upper right')
plt.show()


# In[57]:


print(is_canceled_corr.abs().sort_values(ascending = False)[1:])


# In[58]:


remove_list = ['children','arrival_date_year','babies','stays_in_weekend_nights']


# In[59]:


num_data = [col for col in data1.columns if data1[col].dtype != 'O' and col not in remove_list]
print(num_data)


# In[60]:


cat_data = [col for col in data1.columns if data1[col].dtype == 'O']
print(cat_data)


# **Select some of the important categorical features**

# In[61]:


data1['reservation_status'].value_counts() 
# Due to canceled the model will perform well during traing whereas it won't work in test data set. 
# so we remove/drop this column.


# In[62]:


final = data1.drop(['children','arrival_date_year','babies','stays_in_weekend_nights',
                  'reservation_status', 'country',
                   'arrival_date_week_number', 'arrival_date_day_of_month'], axis =1)
final.head(5)


# In[63]:


final['reservation_status_date'] = pd.to_datetime(final['reservation_status_date'])


# In[64]:


final['year'] = final['reservation_status_date'].dt.year
final['month']= final['reservation_status_date'].dt.month
final['day'] = final['reservation_status_date'].dt.day
final = final.drop(['reservation_status_date'], axis= 1)
final.head(5)


# **Feature Encoding**

# In[65]:


for col in final.columns:
    if (final[col].dtypes =='O'):
        cat_data = final[col]
        print(cat_data.name)


# In[66]:


cat_data = final.select_dtypes(include='O')
cat_data.head(5)


# In[67]:


cat_data['cancelled'] = final['is_canceled']


# In[68]:


def mean_encode(df, col, mean_col):
    df_dict = df.groupby([col])[mean_col].mean().to_dict()
    df[col] = df[col].map(df_dict)
    return df

#for col in cat_data.columns[0:8]:
for col in cat_data.columns:
    if col != 'cancelled': # Exclude the 'cancelled' column from mean encoding
        cat_data = mean_encode(cat_data, col,'cancelled')


# In[69]:


cat_data = cat_data.drop(['cancelled'],axis =1)
cat_data.head(5)


# In[70]:


num_data = final.select_dtypes(include=['float64','int64'])
num_data.head(5)


# In[71]:


final = pd.concat([num_data, cat_data], axis =1)
final.head(5)


# In[72]:


print(num_data.columns)
print(cat_data.columns)
print(final.columns)


# **Find and handle the outlier**

# In[73]:


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


# In[74]:


sns.displot(final['lead_time'], kde =True)


# In[75]:


def handle_outlier(col):
    final[col] = np.log1p(final[col]) 


# In[76]:


handle_outlier('lead_time')


# In[77]:


sns.displot(final['lead_time'], kde =True)


# In[78]:


final['lead_time'] = final['lead_time'].dropna()


# In[79]:


sns.displot(final['lead_time'], kde =True)


# In[80]:


sns.distplot(final['adr'], kde= True)


# In[81]:


final['adr'] = np.log1p(final['adr'])


# In[82]:


final['adr']= final['adr'].dropna()
final['adr'].info()


# In[83]:


final= final.dropna(subset=['adr'])
sns.distplot(final['adr'], kde = True)


# In[84]:


final['adr'].isnull().sum()


# In[85]:


final.info()


# **Delcaring X & y Variables**

# In[86]:


X = final.drop('is_canceled', axis= 1)
y = final['is_canceled']


# **Split the training data set**

# In[88]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,f1_score


# In[89]:


from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state =42)


# In[90]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# **Model Building and Evaluation**

# In[100]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[112]:


from sklearn.pipeline import Pipeline 
models = {'Logistic Regression': LogisticRegression(),
         'Decision Tree Classifier': DecisionTreeClassifier(),
         'Random Forest Classifier': RandomForestClassifier(),
         'Gradient Boosting Classifier': GradientBoostingClassifier(),
         #'KNeighbors Classifier': KNeighborsClassifier(n_neighbors =5),
         'SVM': SVC(),
         'SCGD Classifier': SGDClassifier(),
         'Naive_bayes': GaussianNB()}


# In[113]:


results = {}

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        classification = classification_report(y_test, pred)
        results[name] = {
            'accuracy': accuracy,
            'classification_report': classification
        }
        print(f"{name} - Accuracy: {accuracy:.4f}")
        print(classification)
    except Exception as e:
        print(f"Error with model {name}: {e}")


# #### Conclusion

# **Summary of Findings**    
# In this project, we aimed to predict hotel booking cancellations using various machine learning models. We utilized a dataset comprising 87,388 booking records, each with 25 features. The primary models evaluated included Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Gradient Boosting Classifier, SVM, SCGD Classifier, and Naive Bayes. Here are the key performance metrics of the models:

# Among the models tested, the Random Forest Classifier achieved the highest accuracy (0.9095) and demonstrated a balanced performance in terms of precision, recall, and F1-score. This suggests that ensemble methods like Random Forest are effective for this type of prediction problem. The model's ability to handle a large number of features and its robustness to overfitting contributed to its superior performance.
# 
# The Naive Bayes classifier, however, performed poorly, indicating that its assumptions about feature independence do not hold well for this dataset.
