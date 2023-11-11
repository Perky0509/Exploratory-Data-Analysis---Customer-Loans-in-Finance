
import pandas as pd 
import csv 
import numpy as np

## FULLY COPIED TO PY VERSION ##

loans_for_info = pd.read_csv('tansformed_loan_payments.csv')

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
    

loan = DataFrameInfo(loans_for_info)

categ_list = ['sub_grade', 'grade', 'loan_status']
print(loan.unique_categories(categ_list))

#saving updated file for next class 
loans_for_info.to_csv('loans_for_info.csv')
