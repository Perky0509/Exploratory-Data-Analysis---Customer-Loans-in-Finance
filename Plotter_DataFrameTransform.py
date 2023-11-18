import csv
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import plotly as px
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor 


# ------------------------------------------------------------------------- Transforming Data Frame -------------------------------------------------------------------------- # 
#opening csv file
transformed_loan_payments = pd.read_csv('transformed_loan_payments.csv')

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

#saving file
transformed_loan_payments.to_csv('transformed_loan_payments.csv')
