import csv
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import URL
import yaml


"""this is a class that allows us to extract the data we need to our EDA"""
class RDSDatabaseConnector:
 
    #This is to return the credentials dictionary necessary to create the SQLAlchemy 
    def open_yaml_credentials():
        with open('./credentials.yaml', 'r') as file:
                dictionary_of_credentials = yaml.safe_load(file)    
 
    def __init__(self, dictionary_of_credentials):  
         self.dictionary_of_credentials = dictionary_of_credentials
         self.open_yaml_credentials = open_yaml_credentials()
    
    #This method uses SQLAlchemy and the RDS to connect to the db
    def initialise_engine(self):
        self.dictionary_of_credentials = self.open_yaml_credentials
        connection_string = f"postgresql://{self.dictionary_of_credentials['RDS_USER']}:{self.dictionary_of_credentials['RDS_PASSWORD']}@{self.dictionary_of_credentials['RDS_HOST']}:{self.dictionary_of_credentials['RDS_PORT']}/{self.dictionary_of_credentials['RDS_DATABASE']}"
        engine = create_engine(connection_string)
        return engine

    #Extracting the table we need from the RDS db and using Pandas to view it
    def df_extraction(self):
        extraction_engine = self.initialise_engine()
        df = pd.read_sql_table('loan_payments', extraction_engine)

    #Saving loan_payments df as a csv file to speed up loading on local machine
    def loan_payments_df_as_csv(self):
        extraction_engine = self.initialise_engine()
        df = pd.read_sql_table('loan_payments', extraction_engine)
        df.to_csv('loan_payments.csv', index=False)
        return df


df = RDSDatabaseConnector(open_yaml_credentials())

print(df.loan_payments_df_as_csv())

 




