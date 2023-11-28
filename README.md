# Exploratory Data Analysis - Customer Loans in Finance 

## Table of Contents
- ### Project Description
- ### Installation Instructions
- ### Usage Instructions 
- ### File Structure
- ### Visualisations
- (Visualisations) - Current Loan Status
- (Visualisations) - Loss and Potential Loss
- (Visualisations) - Indicators of Loss
- ### License Information

## Project Description
 - This project is all about performing experimental data analysis on a data table extracted from a database stored in AWS RDS. 
 - There were many learnings from doing this project. It required me to put into practise a wide range of knowledge and skills picked up so far on the AiCore Data Science course. It was also great to see how, using code, tabular data can be manipulated to find all sorts of information, and transformed into easily digestible plots to help everyone understand what the data is saying.  

## Installation Instructions
There are several py libraries that will need to be installed in order to run the code in this repo:
- SQLAlchemy
- PyYaml
- csv
- Pandas
- NumPy
- Scipy
- statsmodels
- Matplotlib
- Sklearn
- Seaborn


## Usage instructions
I have removed all references to print() other than in areas necessary to extract and update the original dataframe and its subsets in order to avoid an overwhelming amount of simultaneous output and confusion. Therefore to see results of analysis and visualisations these will need to be added by the user.

## File Structure of the Project
For this project there is an overall project file, with all the code laid out in one document. Then there are individual files, one per task, to hone in on different data processing, cleaning and analytical elements

## Visualisations
In order to see an initial overview of the analysis in this project here is a collection of visualisations to look out without needing to run the code. 

### Looking at skew and missing values

To start with this is an overview of the skew of each variables


<img width="424" alt="Screenshot 2023-11-10 at 14 26 05" src="https://github.com/Perky0509/Exploratory-Data-Analysis---Customer-Loans-in-Finance/assets/145782195/1f935188-7b33-444b-8dcf-bfe4b2daa99d">



This q-q plot shows that 'total payment inv' is not normally distributed.



<img width="545" alt="Screenshot 2023-11-11 at 23 39 23" src="https://github.com/Perky0509/Exploratory-Data-Analysis---Customer-Loans-in-Finance/assets/145782195/a3b77c30-cb4c-49a9-994b-9db0e7097b6e">



...and this histogram illustrates just how positively skewed 'total_rec_late_fee' was. The columns in this df were almost exclusively positively skewed, but not all as extremely as this.


<img width="569" alt="Screenshot 2023-11-10 at 15 06 28" src="https://github.com/Perky0509/Exploratory-Data-Analysis---Customer-Loans-in-Finance/assets/145782195/ed184138-3d96-496a-b32c-a8cbc1be0067">



This boxcox transformation visual shows the effect of imputing the annual income col with the median. This was chosen as the median is better than the mean at neutralising the impact of outliers.


<img width="285" alt="Screenshot 2023-11-18 at 22 18 42" src="https://github.com/Perky0509/Exploratory-Data-Analysis---Customer-Loans-in-Finance/assets/145782195/97ce3cda-4f47-437a-9cd6-99dacba3bd38">





### Current Loan Status

In this example we are seeing the how much customers are projected to have paid off (in %) in 6 months time. Hence some customers appear to have paid off >100%.


<img width="389" alt="Screenshot 2023-11-18 at 16 39 46" src="https://github.com/Perky0509/Exploratory-Data-Analysis---Customer-Loans-in-Finance/assets/145782195/ff1551f4-03cd-4362-baa1-f0f4bbf46379">



This simply shows that those in the "fully paid" bracket have paid off their loan.

<img width="403" alt="Screenshot 2023-11-18 at 21 52 31" src="https://github.com/Perky0509/Exploratory-Data-Analysis---Customer-Loans-in-Finance/assets/145782195/86fcac8b-1e81-4e0e-ba75-d01948324ab6">


### Indicators of Loss 

Overall the most important indicator of loss is when the customer paid their last installment, followed by when they had to pay their next.

<img width="374" alt="Screenshot 2023-11-18 at 22 03 30" src="https://github.com/Perky0509/Exploratory-Data-Analysis---Customer-Loans-in-Finance/assets/145782195/d9a857ba-8124-4f04-9313-487033adb02b">



This lmplot example shows that for 'potentially charged off' customers (ie those who are late on payments/in their grace period) the potential effect of their last payment date is greater than the threat of the next installment date.



<img width="283" alt="Screenshot 2023-11-18 at 22 03 17" src="https://github.com/Perky0509/Exploratory-Data-Analysis---Customer-Loans-in-Finance/assets/145782195/d332803c-e6d4-47c0-86e2-31b1f36f7b77">


## License information
MIT 
