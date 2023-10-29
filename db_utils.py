import csv
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import URL
import yaml


"""this is a class that allows us to extract the data we need to our EDA"""
class RDSDatabaseConnector:


    def open_yaml_credentials():
        with open('credentials.yaml', 'r') as file:
                yaml_credentials = yaml.safe_load(file)
        print(yaml_credentials)
        

    def open_csv_df():
        with open('loan_payments.csv', 'r') as f:
            loan_payments = csv.reader(f, delimiter= ',')
            for row in loan_payments:
                print(', '.join(row))

     #within the initialisation we are opening the yaml file necessary to open the db
    def __init__(dictionary_of_credentials = open_yaml_credentials()): 
        self.dictionary_of_credentials = dictionary_of_credentials
        self.engine = engine
        
    
    #This method uses SQLAlchemy and the RDS to connect to the db
    def initialise_engine(self, url):
       self.engine = create_engine(url)
       self.engine.connect()

    #Extracting the table we need from the RDS db and using Pandas to view it
    def df_extraction(self):
        loan_payments = pd.read_sql_table('loan_payments', self.engine)
        return loan_payments
    
    #Saving loan_payments df as a csv file to speed up loading on local machine
    def df_as_csv(self):
        self.df_extraction()
        loan_payments.to_csv('loan_payments.csv', index=False) 



