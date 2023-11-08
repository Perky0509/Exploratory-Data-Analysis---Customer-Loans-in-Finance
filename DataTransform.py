#a child class of the RDSDatabaseConnector class
import pandas as pd
import numpy as np 

loan_payments = pd.read_csv('loan_payments.csv')
transformed_loan_payments = loan_payments.copy()


"""Now we are beginning of cleaning and manipulating the loan_payments dataframe in order to make it ready for our calculations. 
 This class focuses on formatting - specifically dtypes and rounding. 
 The attribute of the object is the dataframe we're working with,
 and the methods iterate through a predefined list of applicable columns within said dataframe."""

class  DataTransform():

    def __init__(self, df = pd.DataFrame):
         self.df = df
         #self.series = pd.Series

    #changing float64 values to int where applicable
    def float_to_int(self, cols: list):
         for col in cols:
            self.df[col] = self.df[col].astype(int)
            print(self.df[col].head())
         return self.df[col].info()
    
    #on the other hand some ints may be better represented by float64 dtype - changing int values to float64 
    def int_to_float(self, cols: list):
        for col in cols:
            self.df[col] = self.df[col].astype('float64')
            print(self.df[col].head())
        return self.df[col].info()

    #rounding float64 values to 0dp
    def number_rounding(self, cols: list):
        for col in cols:
            self.df[col] = round(self.df[col], 0)
        return self.df[col].head()
    

    #to format columns where the values correspond to a date, but the dtype is not datetime64.               
    def format_date(self, cols: list):
        for col in cols:
            self.df[col] = pd.to_datetime(self.df[col], format = 'mixed')
        return self.df[col].info()
            
            
    #If the column has fewer than 10 unique values then the column's dtype will be changed to category. This should also save some memory.
    def format_object_categorical(self, cols: list):
        for col in cols:
            self.df[col] =  self.df[col].astype('category')
        return self.df[col].info()

    

        
    


loans = DataTransform(transformed_loan_payments)

columns_to_round = ["loan_amount", "annual_inc", "funded_amount", "funded_amount_inv", "instalment", "total_rec_late_fee", "recoveries", "collection_recovery_fee"]
#print(loans.number_rounding(columns_to_round))

categ_list = ['sub_grade', 'grade', 'loan_status']
#print(loans.format_object_categorical(categ_list))

float_list = ['annual_inc', 'recoveries']
print(loans.float_to_int(float_list))

datetime_list = ['last_payment_date', 'next_payment_date']
#print(loans.format_date(datetime_list))





#correcting a spelling mistake in series name (instalment --> installment)
transformed_loan_payments.rename(columns={'instalment' : 'installment'}).head()




transformed_loan_payments.to_csv('transformed_loan_payments.csv')
transformed_loan_payments.info()
