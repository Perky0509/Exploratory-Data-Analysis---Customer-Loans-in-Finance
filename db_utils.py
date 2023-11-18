import csv
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np 
import pandas as pd
import plotly as px
import seaborn as sns
from scipy import stats
from sqlalchemy import create_engine
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor 
import yaml


# -------------------------------------------------------------------------------- Extracting Data ------------------------------------------------------------------------------------ #
#First of all credentials need to be used (but not exposed) to extract a table from a database in AWS RDS."""

def open_yaml_credentials():
    with open('./credentials.yaml', 'r') as file:
        dictionary_of_credentials = yaml.safe_load(file) 
        return dictionary_of_credentials
        
"""this is the class that allows us to extract the data we need to our EDA"""
class RDSDatabaseConnector:
    
    def __init__(self, dictionary_of_credentials):  
         self.dictionary_of_credentials = dictionary_of_credentials
         self.open_yaml_credentials = open_yaml_credentials()
         self.df = pd.DataFrame
    
    #This method uses SQLAlchemy and the RDS to connect to the db
    def initialise_engine(self):
        self.dictionary_of_credentials = self.open_yaml_credentials
        connection_string = f"postgresql://{self.dictionary_of_credentials['RDS_USER']}:{self.dictionary_of_credentials['RDS_PASSWORD']}@{self.dictionary_of_credentials['RDS_HOST']}:{self.dictionary_of_credentials['RDS_PORT']}/{self.dictionary_of_credentials['RDS_DATABASE']}"
        engine = create_engine(connection_string)
        return engine

    #Extracting the table we need from the RDS db and using Pandas to view it
    def df_extraction(self):
        extraction_engine = self.initialise_engine()
        self.df = pd.read_sql_table('loan_payments', extraction_engine)
        return self.df

    #Saving loan_payments df as a csv file to speed up loading on local machine
    def loan_payments_df_as_csv(self):
        dframe = self.df_extraction()
        dframe.to_csv('loan_payments.csv', index=False)
        return self.df
        

loan_payments = RDSDatabaseConnector(open_yaml_credentials())

print(loan_payments.loan_payments_df_as_csv())

#opening csv file 
transformed_loan_payments = pd.read_csv('loan_payments.csv')

  
# -------------------------------------------------------------------------------- Transforming the Data ----------------------------------------------------------------------------- #

#This class's purpose is to change the dtypes of some columns, as well as to round the float values. It is intended to make the df more readable and ready for manipulation and analysis
class  DataTransform():

    def __init__(self, df = pd.DataFrame):
         self.df = df

    #changing float64 values to int where applicable
    def float_to_int(self, cols: list):
         for col in cols:
            self.df[col] = self.df[col].astype('int')
            print(self.df[col].head())
         return self.df[col].info()
    
    #conversely - changing some int values to float64 where applicable
    def int_to_float(self, cols: list):
        for col in cols:
            self.df[col] = self.df[col].astype('float64')
            print(self.df[col].head())
        return self.df[col].info()

    #rounding float64 values to 0dp
    def number_rounding_2dp(self, cols: list):
        for col in cols:
            self.df[col] = round(self.df[col], 2)
        return self.df[col].head()
    
    #formatting columns where the values correspond to a date, but the dtype is not datetime64.               
    def format_date(self, cols: list):
        for col in cols:
            self.df[col] = pd.to_datetime(self.df[col], format = 'mixed')
        return self.df[col].info() 
            
    #If the column has fewer than 10 unique values (predefined list) then the column's dtype will be changed to category. 
    def format_object_categorical(self, cols: list):
        for col in cols:
            self.df[col] =  self.df[col].astype('category')
        return self.df[col].info()  

loans = DataTransform(transformed_loan_payments)


columns_to_round = ["funded_amount", "funded_amount_inv", "instalment", "total_rec_late_fee", "collection_recovery_fee", "last_payment_amount"]
#print(loans.number_rounding_2dp(columns_to_round))

categ_list = ['term', 'sub_grade', 'grade', 'home_ownership', 'verification_status', 'loan_status', 'payment_plan', 'purpose', 'application_type']
print(loans.format_object_categorical(categ_list))

float_for_int = ['annual_inc', 'dti', 'recoveries']
print(loans.float_to_int(float_for_int))

int_for_float = ["loan_amount"]
print(loans.int_to_float(int_for_float))

datetime_list = ['earliest_credit_line', 'issue_date', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']
print(loans.format_date(datetime_list))

# ------------------------------------------------------------------------------- DataFrame Info ---------------------------------------------------------------------------------- #

"""This class allows the user to find out more about the dataframe, enabling greater understanding of what needs to be altered
and can be done in terms of manipulation and calculations."""

class DataFrameInfo():

    def __init__(self, df = pd.DataFrame):
        self.df = df

    #Returns the dtypes of each column, plus the number of non-null values
    def df_dtypes(self):
        return self.df.info()

    #An overview of median, mean, quartiles and more for all columns
    def statistical_info(self):
        return self.df.describe()
        
    #for every column in the df, if the dtype is 'category' then the output will show what the categories are
    def unique_categories(self, cols: list):
        for col in cols:
            unique_values = len(self.df[col].unique())
        return unique_values
    
    #df's array shape
    def df_shape(self):
        return self.df.shape  

    #Percentage of values that are null in each column
    def null_count(self):
        percentage = self.df.isna().sum() / len(self.df) * 100
        return percentage.round()
    

loan = DataFrameInfo(transformed_loan_payments)


# ------------------------------------------------------------------------- Transforming Data Frame -------------------------------------------------------------------------- # 

"""first we must establish the percentage of missing/null values within each column of the df. Once we've done that we can either delete rows, delete columns or impute. 
This will depend on how high the percentage is (if huge delete col, if not big, but significant stil remove rows, and if tiny impute using mean)"""

#calculating the percentage of null values in the df, shown per column
percent_null_values = round(transformed_loan_payments.isna().sum() / len(transformed_loan_payments) * 100, 1) 
#print(percent_null_values)


#Because the columns with high, but under 65% nulls were key columns (eg next_payment_date - 60.1%) without other cols with similar purpose, the % required to be in this list is 65%
cols = ['mths_since_last_delinq', 'mths_since_last_record']

#remove columns that have a high % of null/missing vals. 
def remove_cols(transformed_loan_payments, cols : list):
    for col in cols:
        transformed_loan_payments = transformed_loan_payments.drop(col, axis= 1)    
    return transformed_loan_payments.info()
remove_cols(transformed_loan_payments, cols)

#calculating the percent of non-null values per row
percent_null_in_rows = transformed_loan_payments.apply(lambda x: round(x.count() / len(transformed_loan_payments.axes[1]) * 100), axis=1)

#calculating how many rows have below 80% non-null values. 
high_null_rows = percent_null_in_rows < 80
print(high_null_rows.value_counts())
#output = 0 so no rows to be deleted 

#saving numeric columns as separate df
numeric_cols = transformed_loan_payments.select_dtypes(include='number')
numeric_cols
numeric_cols.to_csv('numeric_cols.csv')


"""Class to perform manipulations of the table in order to allow for clean and effective analysis. No rows are to be cut so the methods will be to remove columns and impute"""

class DataFrameTransform():

    def __init__(self, df = pd.DataFrame):
        self.df = df
    
    #for normally distributed data
    def impute_mean_data(self, cols: list):
        for col in cols:
           self.df[col] = self.df[col].fillna(self.df[col].mean())
        return self.df[col].info()

    #for category data
    def impute_mode_data(self, cols: list):
        for col in cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        return self.df[col].info()
    
    #for skewed data
    def impute_median_data(self, cols: list):
        for col in cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        return self.df[col].info()    

    #log transformation 
    def log_transformation_and_vis(self, col):
        self.col = self.df[col]
        np.random.seed(0)
        data = self.col
        data_log = np.log(self.col)
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].hist(self.col, edgecolor='black')
        axs[1].hist(data_log, edgecolor='black')
        axs[0].set_title('Original Data')
        axs[1].set_title('Log-Transformed Data')   
    
    #boxcox transformation   
    def box_cox_transformation(self, col):
        self.col = self.df[col]
        plt.figure(figsize = (8, 8))
        data = self.col
        sns.displot(data)
        plt.show()   
        tdata = stats.boxcox(self.col.values.flatten())[0]
        sns.displot(tdata)
        plt.show()   

    #here we remove any value outside a given range. For added visualisation a boxplot is printed before and after the deletion of the outliers.
    def remove_outliers(self, col):
        self.col = self.df[col]
        sns.boxplot(self.col)
        plt.show()
        Q1 = self.col.quantile(0.25)
        Q3 = self.col.quantile(0.75)
        IQR = Q3 - Q1
        self.col = self.df[(self.col >= Q1 - 1.5*IQR) & (self.col <= Q3 + 1.5*IQR)]
        sns.boxplot(self.col)
        plt.title(f'{col} - no_outliers')
        plt.show()
    
loan_df = DataFrameTransform(transformed_loan_payments)

#median chosen because of the nature of the distributions
median_cols = ['int_rate', 'last_payment_amount', 'collections_12_mths_ex_med', 'collections_12_mths_ex_med', 'mths_since_last_major_derog']
print(loan_df.impute_median_data(median_cols))


#using the numeric df created above to pass through a correlation matrix
data_for_matrix = pd.read_csv('numeric_cols.csv').dropna()
corr_matrix = data_for_matrix.corr().abs()
corr_matrix

#the level of high correlation is set to anything above 0.9
high_corr = corr_matrix > 0.9
high_corr #number of high correlations: 38

#using the results of the it is determined that the below columns should be dropped. 
cols_to_drop = ['id', 'total_rec_int', 'policy_code', 'application_type']
transformed_loan_payments = transformed_loan_payments.drop(cols_to_drop, axis=1)

# ------------------------------------------------------------------- Plotting Data Transformations ------------------------------------------------------------------- #


class Plotter():

    def __init__(self, df = pd.DataFrame):
        self.df = df

    def qq_plot(self, col : str): 
        self.col = self.df[col]
        self.qq_plott = sm.qqplot(self.col, scale= 1, line= 'q', fit=True)
        plt.title(f'{col}')
        plt.show()
    
    def boxplot(self, col: str):
        self.col = self.df[col]
        self.boxplot = sns.boxplot(self.col)
        plt.show()

loan_df = Plotter(transformed_loan_payments)


# ------------------------------------------------------------- Current Status of Loans ------------------------------------------------------------------ #
    

#percentage of loans "fully paid" overall:
fully_paid = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Fully Paid')]
len(fully_paid) #28021, so 51.67% 


#payment_inv : funding_inv
ratio_of_funding_inv = round(transformed_loan_payments['total_payment_inv'] / transformed_loan_payments['funded_amount_inv'], 2)
transformed_loan_payments['recovered_ratio_inv'] = ratio_of_funding_inv

#checking with another method
transformed_loan_payments['percentage_repaid_inv'] = round(transformed_loan_payments['total_payment_inv'] / transformed_loan_payments['funded_amount_inv'] * 100, 2)
repaid_inv = transformed_loan_payments['percentage_repaid_inv'] >=1 
paid_back_inv = repaid_inv.sum()
percentage_repaid_on_inv = paid_back_inv / len(transformed_loan_payments) * 100 
percentage_repaid_on_inv #91.02

#payment : funding 
ratio_of_funding = round(transformed_loan_payments['total_payment'] / transformed_loan_payments['funded_amount'])
transformed_loan_payments['recovered_ratio_total'] = ratio_of_funding

#percentage of overal loans amount recovered against total amount funded 
transformed_loan_payments['percentage_repaid_total'] = round(transformed_loan_payments['total_payment'] / transformed_loan_payments['funded_amount'] *100, 2)
repaid_total = transformed_loan_payments['percentage_repaid_total'] >=1 
paid_back_inv = repaid_total.sum()
percentage_repaid_total = round(paid_back_inv / len(transformed_loan_payments) * 100, 2) 
print(percentage_repaid_total) #94.42


unique_loan_status = transformed_loan_payments['loan_status'].unique()

def loan_status_histogram(df, col, loan_statuses):
    for status in loan_statuses:
        subset = df[df['loan_status'] == status]
        plt.figure()
        sns.histplot(subset[col], kde=False)
        plt.title(f'{status} - {col}')
        plt.show()

#histogram to show amount repaid per loan status
loan_status_histogram(transformed_loan_payments, "percentage_repaid_total", unique_loan_status)

#histogram to show inv amount repaid per loan status
loan_status_histogram(transformed_loan_payments, "percentage_repaid_inv", unique_loan_status)


# -------------------------------------------------------------- Six Months Projection ------------------------------------------------------------------- #

#Subset
relevant_customers = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains("Current|Late|Grace")]
customers_still_paying = relevant_customers

#calculating what the percentage of customers will have paid off their loan within 6m 
customers_still_paying['paid_6m'] = round(((customers_still_paying["total_payment"] + (customers_still_paying["instalment"] * 6)) / customers_still_paying["funded_amount"]) * 100, 2)
print(len(customers_still_paying['paid_6m']))

paid_by_6m = customers_still_paying['paid_6m'] >= 1
sum_of_paid_6m = paid_by_6m.sum()
sum_of_paid_6m #19095

total_percentage_paid_6m = round(sum_of_paid_6m / len(customers_still_paying) * 100, 2)
total_percentage_paid_6m #94.44%

#histogram how many of each currently paying loan status will have paid off the loan 
def paid_6m_histogram(df, col):
    for status in unique_loan_status:
        subset = df[df['loan_status'] == status]
        plt.figure()
        sns.histplot(subset[col], kde=False)
        plt.title(f'{status} - {col}')
        plt.show()

paid_6m_histogram(customers_still_paying, "paid_6m")


#------------------------------------------------------------- Charged Off % and Amount Paid -------------------------------------------------------------------------------#


#percentage of charged-off loans historically; 
tansformed_loan_payments = pd.read_csv('tansformed_loan_payments.csv')
charged_off = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Charged Off')]
number_of_charged_off = len(charged_off)
number_of_charged_off #5939

total_loan_status = len(transformed_loan_payments['loan_status'].dropna())
percentage_charged_off = number_of_charged_off / total_loan_status * 100
percentage_charged_off #10.27%


#calculating the amount paid towards these loans, both per row and as a total. This is done by extracting values in 'total_payment' from rows where the status is charged off. 
#total 
amount_paid_total = round(charged_off['total_payment'].sum(), 2)
amount_paid_total #£39,247,128.38

#per column
amount_paid_per_col = round(charged_off['total_payment'], 2)
amount_paid_per_col

#------------------------------------------------------------------ Charged Off Projected Loss ----------------------------------------------------------------------------#

transformed_loan_payments = pd.read_csv('transformed_loan_payments.csv')

#First we need to change dtypes of term (category) float64 within original df. 
# (N.B. loan_amount and int_rate are already dtype float64 

#removing str from 'term' values 
transformed_loan_payments['term'] = transformed_loan_payments['term'].str.replace(r'\D', '', regex=True)
#changing new col values to dtype float64
transformed_loan_payments['term'] = transformed_loan_payments['term'].astype(float)


#In order to find the percentage loss, need to calculate the overall amount the customer would pay back over time (the term * int_rate * loan amount) and then divide the amount that's already been paid by that. 
#N.B. I've used 'loan_amount' rather than funded_amount(_inv) because loan amount is shown as the overall amount owed. 

#multiplying the total loan amount by the interest rate over term months 
loan_amount_incl_int_rate = transformed_loan_payments['term'] + (1 + transformed_loan_payments['int_rate'] / 100) * transformed_loan_payments['loan_amount']

#creating a new col of the multiplied values to our charged-off only df
transformed_loan_payments['loan_amount_incl_int_rate'] = transformed_loan_payments['term'] + (1 + transformed_loan_payments['int_rate'] / 100) * transformed_loan_payments['loan_amount']

#dividing total_payment by these values * 100 in order to get percentage
percentage_loss = transformed_loan_payments['total_payment'] / transformed_loan_payments['loan_amount_incl_int_rate'] * 100
percentage_loss
#creating a new column to show each row's percentage loss
transformed_loan_payments['percentage_loss'] = percentage_loss

#calculating and creating a new column to show each row's financial loss
money_lost = round(transformed_loan_payments['loan_amount_incl_int_rate'] - transformed_loan_payments['total_payment'], 2)
transformed_loan_payments['money_lost'] = money_lost

charged_off_potential_loss = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Charged Off')]
#seeing the overall financial loss to the company
total_loss_charged_off = round(charged_off_potential_loss['loan_amount_incl_int_rate'].sum() - charged_off_potential_loss['total_payment'].sum(), 2)
total_loss_charged_off #£37,143,206.46


#calculating amount of revenue company could have made if loans weren't charged off 
potential_revenue_charged_off = charged_off_potential_loss['loan_amount_incl_int_rate'].sum()
potential_revenue_charged_off #£76,390,334.84


# -------------------------------------------------------------------- Potential Loss ----------------------------------------------------------------------------------- #


#reminder of the unique values in loan status
unique_loan_status
"""['Current', 'Fully Paid', 'Charged Off', 'Late (31-120 days)',
       'In Grace Period', 'Late (16-30 days)', 'Default',
       'Does not meet the credit policy. Status:Fully Paid',
       'Does not meet the credit policy. Status:Charged Off'],
      dtype=object)"""

#counting number of customers late on their payments. I have included those who are in their Grace Period here as they've still missed a payment
late_customers = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Late | Grace')]

number_late_customers = len(late_customers)
number_late_customers #951


#calculating what percentage of customers are late on their payments
proportion_late = round(number_late_customers / len(transformed_loan_payments['loan_status']) * 100, 2)
proportion_late #1.75%


#total amount the 'late' loans are worth
total_late_loan_amount = round(late_customers['loan_amount_incl_int_rate'].sum(), 2)
total_late_loan_amount #£13,843,716.83


#loss to revenue if late --> charged off (not incl. already charged off)
potential_loss_late = round(total_late_loan_amount - late_customers['total_payment'].sum())
potential_loss_late #£3,024,147


#loss of revenue of late and charged off customers 
potential_loss_late_charged_off = potential_loss_late + total_loss_charged_off
potential_loss_late_charged_off #£40,845,811.46


#creating a sebset to show data for customers who have already defaulted
default = transformed_loan_payments[transformed_loan_payments['loan_status'] == 'Default']

potential_revenue_default = round(default['loan_amount_incl_int_rate'].sum(), 2)
potential_revenue_default #£781,672.94

loss_from_default = round((default['loan_amount_incl_int_rate'] - default['total_payment']).sum(), 2)
loss_from_default #333,486.22

# % of overall revenue that late/charged off and default customers present (total value of loans)
overall_revenue_proportion = round((potential_revenue_charged_off + total_late_loan_amount + potential_revenue_default) / transformed_loan_payments['loan_amount_incl_int_rate'].sum() * 100, 2)   
overall_revenue_proportion #13.32%    
                      
# % of overall revenue that late/charged off and default customers present (potential loss as % of total revenue)
loss_choff_late_default = (loss_from_default + potential_loss_late_charged_off) / transformed_loan_payments['loan_amount_incl_int_rate'].sum() * 100
loss_choff_late_default #6.02%


# --------------------------------------------------------------------- Indicators of Loss ---------------------------------------------------------------------------------- #

#counting number of customers late on their payments. I have included those who are in their Grace Period here as they've still missed a payment
late_customers = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Late | Grace')]

number_late_customers = len(late_customers)
number_late_customers #951


#calculating what percentage of customers are late on their payments
proportion_late = round(number_late_customers / len(transformed_loan_payments['loan_status']) * 100, 2)
proportion_late #1.75%


#total amount the 'late' loans are worth
total_late_loan_amount = round(late_customers['loan_amount_incl_int_rate'].sum(), 2)
total_late_loan_amount #£13,843,716.83


#loss to revenue if late --> charged off (not incl. already charged off)
potential_loss_late = round(total_late_loan_amount - late_customers['total_payment'].sum())
potential_loss_late #£3,024,147


#loss of revenue of late and charged off customers 
potential_loss_late_charged_off = potential_loss_late + total_loss_charged_off
potential_loss_late_charged_off #£40,845,811.46


#creating a sebset to show data for customers who have already defaulted
default = transformed_loan_payments[transformed_loan_payments['loan_status'] == 'Default']

potential_revenue_default = round(default['loan_amount_incl_int_rate'].sum(), 2)
potential_revenue_default #£781,672.94

loss_from_default = round((default['loan_amount_incl_int_rate'] - default['total_payment']).sum(), 2)
loss_from_default #333,486.22

# % of overall revenue that late/charged off and default customers present (total value of loans)
overall_revenue_proportion = round((potential_revenue_charged_off + total_late_loan_amount + potential_revenue_default) / transformed_loan_payments['loan_amount_incl_int_rate'].sum() * 100, 2)   
overall_revenue_proportion #13.32%    
                      
# % of overall revenue that late/charged off and default customers present (potential loss as % of total revenue)
loss_choff_late_default = (loss_from_default + potential_loss_late_charged_off) / transformed_loan_payments['loan_amount_incl_int_rate'].sum() * 100
loss_choff_late_default #6.02%

#counting number of customers late on their payments. I have included those who are in their Grace Period here as they've still missed a payment
late_customers = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Late | Grace')]

number_late_customers = len(late_customers)
number_late_customers #951


#calculating what percentage of customers are late on their payments
proportion_late = round(number_late_customers / len(transformed_loan_payments['loan_status']) * 100, 2)
proportion_late #1.75%


#total amount the 'late' loans are worth
total_late_loan_amount = round(late_customers['loan_amount_incl_int_rate'].sum(), 2)
total_late_loan_amount #£13,843,716.83


#loss to revenue if late --> charged off (not incl. already charged off)
potential_loss_late = round(total_late_loan_amount - late_customers['total_payment'].sum())
potential_loss_late #£3,024,147


#loss of revenue of late and charged off customers 
potential_loss_late_charged_off = potential_loss_late + total_loss_charged_off
potential_loss_late_charged_off #£40,845,811.46


#creating a sebset to show data for customers who have already defaulted
default = transformed_loan_payments[transformed_loan_payments['loan_status'] == 'Default']

potential_revenue_default = round(default['loan_amount_incl_int_rate'].sum(), 2)
potential_revenue_default #£781,672.94

loss_from_default = round((default['loan_amount_incl_int_rate'] - default['total_payment']).sum(), 2)
loss_from_default #333,486.22

# % of overall revenue that late/charged off and default customers present (total value of loans)
overall_revenue_proportion = round((potential_revenue_charged_off + total_late_loan_amount + potential_revenue_default) / transformed_loan_payments['loan_amount_incl_int_rate'].sum() * 100, 2)   
overall_revenue_proportion #13.32%    
                      
# % of overall revenue that late/charged off and default customers present (potential loss as % of total revenue)
loss_choff_late_default = (loss_from_default + potential_loss_late_charged_off) / transformed_loan_payments['loan_amount_incl_int_rate'].sum() * 100
loss_choff_late_default #6.02%



# ------------------------ Indicators for all lapsed ------------------- #

#subset
dfs_for_indicator_subset = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Late | Grace | Charged Off | Default')]
indicator_subset = dfs_for_indicator_subset

#changing str value columns to number codes
for column in indicator_subset.select_dtypes(['object']):
    indicator_subset[column] = indicator_subset[column].factorize()[0]

#chosen variables to check 
corr_indicator_subset = indicator_subset[['loan_amount', 'term', 'int_rate', 'instalment', 'annual_inc', 'dti', 'loan_status', 'home_ownership', 'purpose', 'total_payment', 'last_payment_date', 'next_payment_date', 'earliest_credit_line', 'total_accounts', 'open_accounts', 'mths_since_last_delinq', 'delinq_2yrs']]

#correlation matrix
corr_matrix = corr_indicator_subset.corr(method= 'kendall')
print(corr_matrix)
#largest indicator is last payment date (0.59), followed by next payment date (0.15). The third biggest indicator was annual income, although barely a correlation (-0.04)

#creating heatmap to show likelihood of each chosen variable being an indicator of loss 
cmap = sns.diverging_palette(20, 220, n=200)
heatmap = sns.heatmap(corr_matrix, cmap=cmap, square= True)
heatmap.set_title( "Potential Indicators of Loss")


# ----------------------------------------- visualising indicators -------------------------------------- #

#last_payment_date
scatter_last_payment = indicator_subset[['loan_status', 'last_payment_date']]
sns.lmplot(x="loan_status", y="last_payment_date", data=scatter_last_payment, line_kws={'color': 'red'})

#next_payment_date
scatter_next_payment = indicator_subset[['loan_status', 'next_payment_date']]
sns.lmplot(x="loan_status", y="next_payment_date", data=scatter_next_payment, line_kws={'color': 'red'})

#annual_income - mainly showing the lack of correlation in comparison to the above two
scatter_annual_inc = indicator_subset[['loan_status', 'annual_inc']]
sns.lmplot(x="loan_status", y="annual_inc", data=scatter_annual_inc, line_kws={'color': 'green'})


# -------------------------------- Indicators when Already Charged Off -------------------------------- #

#subset
loan_status_co_indicator_subset = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Charged Off')]
co_indicator_subset = loan_status_co_indicator_subset

for column in co_indicator_subset.select_dtypes(['object']):
    co_indicator_subset[column] = co_indicator_subset[column].factorize()[0]

#chosen variables to check 
corr_co_indicator_subset = co_indicator_subset[['loan_amount', 'term', 'int_rate', 'instalment', 'annual_inc', 'dti', 'loan_status', 'home_ownership', 'purpose', 'total_payment', 'last_payment_date', 'next_payment_date', 'earliest_credit_line', 'total_accounts', 'open_accounts', 'mths_since_last_delinq', 'delinq_2yrs']]

#correlation matrix
co_corr_matrix = corr_co_indicator_subset.corr(method= 'kendall')
print(co_corr_matrix)
#by far the largest indicator is the next payment date (0.98). After that it's the term length (-0.12). None of the other variables are of particular note. 
#This makes sense as being charged off is based around ability to make payments

#creating heatmap to show likelihood of each chosen variable being an indicator of loss 
cmap = sns.diverging_palette(20, 220, n=200)
heatmap = sns.heatmap(co_corr_matrix, cmap=cmap, square= True)
heatmap.set_title( "Indicators of Loss - Already Charged Off")

# ----------------------------------------- visualising indicators -------------------------------------- #

#next_payment_date
scatter_next_payment = co_indicator_subset[['loan_status', 'next_payment_date']]
sns.lmplot(x="loan_status", y="next_payment_date", data=scatter_next_payment, line_kws={'color': 'green'})

#last_payment_date
scatter_term = indicator_subset[['loan_status', 'term']]
sns.lmplot(x="loan_status", y="term", data=scatter_term, line_kws={'color': 'green'})

# ------------------------------------------ Potentially Charged Off ------------------------------------- #

#subset
loan_status_late_indicator_subset = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Late | Grace')]
late_indicator_subset = loan_status_late_indicator_subset

for column in late_indicator_subset.select_dtypes(['object']):
    late_indicator_subset[column] = late_indicator_subset[column].factorize()[0]

#chosen variables to check 
corr_late_indicator_subset = late_indicator_subset[['loan_amount', 'term', 'int_rate', 'instalment', 'annual_inc', 'dti', 'loan_status', 'home_ownership', 'purpose', 'total_payment', 'last_payment_date', 'next_payment_date', 'earliest_credit_line', 'total_accounts', 'open_accounts', 'mths_since_last_delinq', 'delinq_2yrs']]

#correlation matrix
late_corr_matrix = corr_late_indicator_subset.corr(method= 'kendall')
print(late_corr_matrix)
#The last payment date is the strongest indicator at 0.60. The next most correlated variable is the next payment date, 0.15. The third most, and last of note is total payment at -0.1. 
#It appears that the most likely problem that could lead to these customers becoming charged off is the weight of the payment schedule.

#creating heatmap to show likelihood of each chosen variable being an indicator of loss 
cmap = sns.diverging_palette(20, 220, n=200)
heatmap = sns.heatmap(late_corr_matrix, cmap=cmap, square= True)
heatmap.set_title( "Indicators of Loss - Potentially Charged Off")


# ----------------------------------------- visualising indicators --------------------------------------------- #

#last_payment_date
scatter_last_payment = late_indicator_subset[['loan_status', 'last_payment_date']]
sns.lmplot(x="loan_status", y="last_payment_date", data=scatter_last_payment, line_kws={'color': 'purple'})

#next_payment_date
scatter_next_payment = late_indicator_subset[['loan_status', 'next_payment_date']]
sns.lmplot(x="loan_status", y="next_payment_date", data=scatter_next_payment, line_kws={'color': 'purple'})

#total_payment
scatter_total_payment = late_indicator_subset[['loan_status', 'total_payment']]
sns.lmplot(x="loan_status", y="total_payment", data=scatter_total_payment, line_kws={'color': 'purple'})


#saving a new csv copy
transformed_loan_payments.to_csv('transformed_visualised_payments.csv')
