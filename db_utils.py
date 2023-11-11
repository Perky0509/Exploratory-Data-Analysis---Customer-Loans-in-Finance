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

"""EXTRACTING DATASET

First of all credentials need to be used (but not exposed) to extract a table from a database in AWS RDS."""

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

  
"""TRANSORMING RAW, YET TO BE CLEANED DATA

This class's purpose is to change the dtypes of some columns, as well as to round the float values. It is intended to make the df more readable and ready for manipulation and analysis"""
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

columns_to_round = ["loan_amount", "funded_amount", "funded_amount_inv", "instalment", "total_rec_late_fee", "collection_recovery_fee", "last_payment_amount"]
print(loans.number_rounding_2dp(columns_to_round))

categ_list = ['term', 'sub_grade', 'grade', 'home_ownership', 'verification_status', 'loan_status', 'payment_plan', 'purpose', 'application_type']
print(loans.format_object_categorical(categ_list))

float_for_int = ['annual_inc', 'dti', 'recoveries']
print(loans.float_to_int(float_for_int))

datetime_list = ['earliest_credit_line', 'issue_date', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']
print(loans.format_date(datetime_list))


#correcting a spelling mistake in series name (instalment --> installment)
spelling = transformed_loan_payments.rename(columns={'instalment' : 'installment'})
print(spelling)





"""GAINING INSIGHT INTO THE DF

This class allows the user to find out more about the dataframe, enabling greater understanding of what needs to be altered
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

print(loan.df_dtypes())
print(loan.statistical_info())
#example categ_list
categ_list = ['term', 'sub_grade', 'grade', 'home_ownership', 'verification_status', 'loan_status', 'payment_plan', 'purpose', 'application_type']
print(loan.unique_categories(categ_list))
print(loan.df_shape())
print(loan.null_count())




"""MISSING VALS: 

first we must establish the percentage of missing/null values within each column of the df. Once we've done that we can either delete rows, delete columns or impute. 
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
        plt.title('no_outliers')
        plt.show()
    
loan_df = DataFrameTransform(transformed_loan_payments)

#median chosen because of the nature of the distributions
median_cols = ['int_rate', 'last_payment_amount', 'collections_12_mths_ex_med', 'collections_12_mths_ex_med', 'mths_since_last_major_derog']
print(loan_df.impute_median_data(median_cols))
print(loan_df.box_cox_transformation('annual_inc'))
print(loan_df.log_transformation_and_vis('annual_inc'))
print(loan_df.remove_outliers(['total_rec_late_fee']))


#using the numeric df created above to pass through a correlation matrix
data_for_matrix = pd.read_csv('numeric_cols.csv').dropna()
corr_matrix = data_for_matrix.corr().abs()
corr_matrix
#the level of high correlation is set to anything above 0.9
high_corr = corr_matrix > 0.9
print(high_corr)
#using the results of the it is determined that the below columns should be dropped. 
cols_to_drop = ['id', 'funded_amount', 'instalment', 'total_rec_int']
transformed_loan_payments = transformed_loan_payments.drop(cols_to_drop, axis=1)
transformed_loan_payments.head()



"""now we move onto creating a class that illustrates the data tranformation with regards to imputing missing vals"""


class Plotter():

    def __init__(self, df = pd.DataFrame):
        self.df = df

    def qq_plot(self, col : str): 
        self.col = self.df[col]
        self.qq_plott = sm.qqplot(self.col, scale= 1, line= 'q', fit=True)
        plt.title('self.col')
        plt.show()
    
    def boxplot(self, col: str):
        self.col = self.df[col]
        self.boxplot = sns.boxplot(self.col)
        plt.show()


histogram = ['int_rate']

loan_df = Plotter(transformed_loan_payments)

#testing code
print(loan_df.qq_plot(['total_rec_late_fee']))
print(loan_df.boxplot(['total_rec_late_fee']))


#saving a new csv copy
transformed_loan_payments.to_csv('transformed_visualised_payments.csv')
