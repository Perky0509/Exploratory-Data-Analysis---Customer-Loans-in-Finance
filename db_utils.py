import yaml
import pandas as pd
from sqlalchemy import create_engine
import csv

def open_yaml():
    with open('credentials.yaml', 'r') as file:
         credentials = yaml.safe_load(file)
    print(credentials)

def df_as_csv(self, loan_payments):
        loan_payments.to_csv('loan_payments.csv', index=False) 



"""this is a class that will enable extraction of the data in order to perform EDA"""
class RDSDatabaseConnector:

     #within the initialisation we are opening the yaml file necessary to open the db
    def __init__(dictionary_of_credentials = credentials): 
        self.dictionary_of_credentials = dictionary_of_credentials
    
    #This method uses SQLAlchemy and the RDS to connect to the db
    def initialise_engine(self):
       engine = create_engine('')
       engine.connect()

    #Extracting the table we need from the RDS db and using Pandas to view it
    def df_extraction(self):
        loan_payments = pd.read_sql_table('loan_payments', self.engine)
        return loan_payments
    
    #Saving loan_payments df as a csv file to speed up loading on local machine
    def df_as_csv(self):
        self.table_extraction()
        loan_payments.to_csv('loan_payments.csv', index=False) 

