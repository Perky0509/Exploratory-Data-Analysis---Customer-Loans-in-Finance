import pandas as pd
import plotly as px
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import csv

df_transform = pd.read_csv('loans_for_info.csv')


"""first we must establish the percentage of missing/null values within each column of the df. Once we've done that we can either delete rows, delete columns or impute. 
This will depend on how high the percentage is (if huge delete col, if not big, but significant stil remove rows, and if tiny impute using mean)"""

#calculating the percentage of null values in the df, shown per column
percent_null_values = round(df_transform.isna().sum() / len(df_transform) * 100, 1) 
#print(percent_null_values)


#Because the columns with high, but under 65% nulls were key columns (eg next_payment_date - 60.1%) without other cols with similar purpose, the % required to be in this list is 65%
cols = ['mths_since_last_delinq', 'mths_since_last_record']

#remove columns that have a high % of null/missing vals. 
def remove_cols(df_transform, cols : list):
    for col in cols:
        test_df = df_transform.drop(col, axis= 1)    
    return df_transform.info()
remove_cols(df_transform, cols)

#calculating the percent of non-null values per row
percent_null_in_rows = df_transform.apply(lambda x: round(x.count() / len(df_transform.axes[1]) * 100), axis=1)

#calculating how many rows have below 80% non-null values. 
high_null_rows = percent_null_in_rows < 80
print(high_null_rows.value_counts())
#output = 0 so no rows to be deleted 


"""Class to perform manipulations of the table in order to allow for clean and effective analysis. No rows are to be cut so the methods will be to remove columns and impute"""

class DataFrameTransform():

    def __init__(self, df = pd.DataFrame):
        self.df = df
    
    #for normally distributed data
    def impute_mean_data(self, cols: list):
        for col in cols:
            self.df[col].fillna(self.df[col].mean())
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

loan_df = DataFrameTransform(df_transform)

#eg to test
mean_cols = ['int_rate', 'funded_amount']
print(loan_df.impute_mean_data(mean_cols))


df_transform.to_csv('df_transform.csv')



"""now we move onto creating a class that illustrates the data tranformation with regards to imputing missing vals"""

plotting = pd.read_csv('df_transform.csv')


class Plotter():

    def __init__(self, df = pd.DataFrame):
        self.df = df

    def qq_plot(self, col : str): 
        self.col = self.df[col]
        self.qq_plott = sm.qqplot(self.col, scale= 1, line= 'q', fit=True)
        pyplot.show()

    def show_skew_plot(self, col: str):
        self.col = col
        self.df[self.col].hist(bins= 25)
        print(f"The {self.col} column has a skew of {(self.df[self.col].skew())}")


histogram = ['int_rate']

loan_df = Plotter(plotting)

#testing code
print(loan_df.qq_plot(['total_rec_late_fee']))
print(loan_df.show_skew_plot('total_rec_late_fee'))


#saving a new csv copy
plotting.to_csv('plotting.csv')