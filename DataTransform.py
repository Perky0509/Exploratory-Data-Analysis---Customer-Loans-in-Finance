# -------------------------------------------------------------------------------- Transforming the Data ----------------------------------------------------------------------------- #
from datetime import datetime 
import numpy as np 
import pandas as pd


#opening csv file 
transformed_loan_payments = pd.read_csv('transformed_loan_payments.csv')

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

transformed_loan_payments.to_csv('transformed_loan_payments.csv')