# ------------------------------------------------------------- Current Status of Loans ------------------------------------------------------------------ #
        

#percentage of loans "fully paid" overall:
fully_paid = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Fully Paid')]
len(fully_paid)
#28021, so 51.67% 


#payment_inv : funding_inv
ratio_of_funding_inv = round(transformed_loan_payments['total_payment_inv'] / transformed_loan_payments['funded_amount_inv'], 2)
transformed_loan_payments['recovered_ratio_inv'] = ratio_of_funding_inv
#percentage of overal loans amount recovered against the investor funding
percentrage_repaid_inv = round(transformed_loan_payments['total_payment_inv'].sum() / transformed_loan_payments['funded_amount_inv'].sum() * 100, 2)
percentrage_repaid_inv #91.02


transformed_loan_payments['percentage_repaid_inv'] = round(transformed_loan_payments['total_payment_inv'] / transformed_loan_payments['funded_amount_inv'] * 100, 2)
repaid_inv = transformed_loan_payments['percentage_repaid_inv'] >=1 
paid_back_inv = repaid_inv.sum()
percentage_repaid_on_inv = paid_back_inv / len(transformed_loan_payments) * 100 
percentrage_repaid_inv #91.02

#payment : funding 
ratio_of_funding = round(transformed_loan_payments['total_payment'] / transformed_loan_payments['funded_amount'])
transformed_loan_payments['recovered_ratio_total'] = ratio_of_funding
#percentage of overal loans amount recovered against total amount funded 
percentrage_repaid= round(transformed_loan_payments['total_payment'].sum() / transformed_loan_payments['funded_amount'].sum() *100, 2) 
percentrage_repaid #96.66
transformed_loan_payments['percentage_repaid_total'] = round(transformed_loan_payments['total_payment'] / transformed_loan_payments['funded_amount'] *100, 2)
repaid_total = transformed_loan_payments['percentage_repaid_total'] >=1 
paid_back_inv = repaid_total.sum()
percentage_repaid_total = round(paid_back_inv / len(transformed_loan_payments) * 100, 2) 
print(percentage_repaid_total) #94.42



# -------------------------------------------------------------- Six Months Projection ------------------------------------------------------------------- #

unique_loan_status = transformed_loan_payments['loan_status'].unique()

def loan_status_histogram(df, col, loan_statuses):
    for status in loan_statuses:
        subset = df[df['loan_status'] == status]
        plt.figure()
        sns.histplot(subset[col], kde=False)
        plt.title(f'{status} - {col}')
        plt.show()

#histogram to show amount repaid per loan status
loan_status_histogram(transformed_loan_payments, "percentage_repaid_total", unique_loan_status)

#histogram to show inv amount repaid per loan status
loan_status_histogram(transformed_loan_payments, "percentage_repaid_inv", unique_loan_status)

#Subset
customers_still_paying = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Current | Late | Grace')]


#calculating what the percentage of customers will have paid off their loan within 6m 
customers_still_paying['paid_6m'] = round(((customers_still_paying["total_payment"] + (customers_still_paying["instalment"] * 6)) / customers_still_paying["funded_amount"]) * 100, 2)


paid_by_6m = customers_still_paying['paid_6m'] >= 1
sum_of_paid_6m = paid_by_6m.sum()
sum_of_paid_6m #249

total_percentage_paid_6m = round(sum_of_paid_6m / len(customers_still_paying) * 100, 2)
total_percentage_paid_6m #93.96%

#histogram how many of each currently paying loan status will have paid off the loan 
def paid_6m_histogram(df, col):
    for status in unique_loan_status:
        subset = df[df['loan_status'] == status]
        plt.figure()
        sns.histplot(subset[col], kde=False)
        plt.title(f'{status} - {col}')
        plt.show()

paid_6m_histogram(customers_still_paying, "paid_6m")


#------------------------------------------------------------- Charged Off % and Amount Paid -------------------------------------------------------------------------------#

#percentage of charged-off loans historically; 
tansformed_loan_payments = pd.read_csv('tansformed_loan_payments.csv')
charged_off = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Charged Off')]
number_of_charged_off = len(charged_off)
number_of_charged_off #5939

total_loan_status = len(transformed_loan_payments['loan_status'].dropna())
percentage_charged_off = number_of_charged_off / total_loan_status * 100
percentage_charged_off #10.27%


#calculating the amount paid towards these loans, both per row and as a total. This is done by extracting values in 'total_payment' from rows where the status is charged off. 
#total 
amount_paid_total = round(charged_off['total_payment'].sum(), 2)
amount_paid_total #£39,247,128.38

#per column
amount_paid_per_col = round(charged_off['total_payment'], 2)
amount_paid_per_col

#------------------------------------------------------------------ Charged Off Projected Loss ----------------------------------------------------------------------------#

transformed_loan_payments = pd.read_csv('transformed_loan_payments.csv')

#First we need to change dtypes of term (category) float64 within original df. 
# (N.B. loan_amount and int_rate are already dtype float64 

#removing str from 'term' values 
transformed_loan_payments['term'] = transformed_loan_payments['term'].str.replace(r'\D', '', regex=True)
#changing new col values to dtype float64
transformed_loan_payments['term'] = transformed_loan_payments['term'].astype(float)


#In order to find the percentage loss, need to calculate the overall amount the customer would pay back over time (the term * int_rate * loan amount) and then divide the amount that's already been paid by that. 
#N.B. I've used 'loan_amount' rather than funded_amount(_inv) because loan amount is shown as the overall amount owed. 

#multiplying the total loan amount by the interest rate over term months 
loan_amount_incl_int_rate = transformed_loan_payments['term'] + (1 + transformed_loan_payments['int_rate'] / 100) * transformed_loan_payments['loan_amount']

#creating a new col of the multiplied values to our charged-off only df
transformed_loan_payments['loan_amount_incl_int_rate'] = transformed_loan_payments['term'] + (1 + transformed_loan_payments['int_rate'] / 100) * transformed_loan_payments['loan_amount']

#dividing total_payment by these values * 100 in order to get percentage
percentage_loss = transformed_loan_payments['total_payment'] / transformed_loan_payments['loan_amount_incl_int_rate'] * 100
percentage_loss
#creating a new column to show each row's percentage loss
transformed_loan_payments['percentage_loss'] = percentage_loss

#calculating and creating a new column to show each row's financial loss
money_lost = round(transformed_loan_payments['loan_amount_incl_int_rate'] - transformed_loan_payments['total_payment'], 2)
transformed_loan_payments['money_lost'] = money_lost

charged_off_potential_loss = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Charged Off')]
#seeing the overall financial loss to the company
total_loss_charged_off = round(charged_off_potential_loss['loan_amount_incl_int_rate'].sum() - charged_off_potential_loss['total_payment'].sum(), 2)
total_loss_charged_off #£37,143,206.46


#calculating amount of revenue company could have made if loans weren't charged off 
potential_revenue_charged_off = charged_off_potential_loss['loan_amount_incl_int_rate'].sum()
potential_revenue_charged_off #£76,390,334.84


# -------------------------------------------------------------------- Potential Loss ----------------------------------------------------------------------------------- #


#finding the unique values in loan status
unique_loan_status
"""['Current', 'Fully Paid', 'Charged Off', 'Late (31-120 days)',
       'In Grace Period', 'Late (16-30 days)', 'Default',
       'Does not meet the credit policy. Status:Fully Paid',
       'Does not meet the credit policy. Status:Charged Off'],
      dtype=object)"""

#counting number of customers late on their payments. I have included those who are in their Grace Period here as they've still missed a payment
late_customers = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Late | Grace')]

number_late_customers = len(late_customers)
number_late_customers #951


#calculating what percentage of customers are late on their payments
proportion_late = round(number_late_customers / len(transformed_loan_payments['loan_status']) * 100, 2)
proportion_late #1.75%


#total amount the 'late' loans are worth
total_late_loan_amount = round(late_customers['loan_amount_incl_int_rate'].sum(), 2)
total_late_loan_amount #£13,843,716.83


#loss to revenue if late --> charged off (not incl. already charged off)
potential_loss_late = round(total_late_loan_amount - late_customers['total_payment'].sum())
potential_loss_late #£3,024,147


#loss of revenue of late and charged off customers 
potential_loss_late_charged_off = potential_loss_late + total_loss_charged_off
potential_loss_late_charged_off #£40,845,811.46


#creating a sebset to show data for customers who have already defaulted
default = transformed_loan_payments[transformed_loan_payments['loan_status'] == 'Default']

potential_revenue_default = round(default['loan_amount_incl_int_rate'].sum(), 2)
potential_revenue_default #£781,672.94

loss_from_default = round((default['loan_amount_incl_int_rate'] - default['total_payment']).sum(), 2)
loss_from_default #333,486.22

# % of overall revenue that late/charged off and default customers present (total value of loans)
overall_revenue_proportion = round((potential_revenue_charged_off + total_late_loan_amount + potential_revenue_default) / transformed_loan_payments['loan_amount_incl_int_rate'].sum() * 100, 2)   
overall_revenue_proportion #13.32%    
                      
# % of overall revenue that late/charged off and default customers present (potential loss as % of total revenue)
loss_choff_late_default = (loss_from_default + potential_loss_late_charged_off) / transformed_loan_payments['loan_amount_incl_int_rate'].sum() * 100
loss_choff_late_default #6.02%


# --------------------------------------------------------------------- Indicators of Loss ---------------------------------------------------------------------------------- #

 

#finding the unique values in loan status
unique_loan_status = transformed_loan_payments['loan_status'].unique()
unique_loan_status
"""['Current', 'Fully Paid', 'Charged Off', 'Late (31-120 days)',
       'In Grace Period', 'Late (16-30 days)', 'Default',
       'Does not meet the credit policy. Status:Fully Paid',
       'Does not meet the credit policy. Status:Charged Off'],
      dtype=object)"""

#counting number of customers late on their payments. I have included those who are in their Grace Period here as they've still missed a payment
late_customers = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Late | Grace')]

number_late_customers = len(late_customers)
number_late_customers #951


#calculating what percentage of customers are late on their payments
proportion_late = round(number_late_customers / len(transformed_loan_payments['loan_status']) * 100, 2)
proportion_late #1.75%


#total amount the 'late' loans are worth
total_late_loan_amount = round(late_customers['loan_amount_incl_int_rate'].sum(), 2)
total_late_loan_amount #£13,843,716.83


#loss to revenue if late --> charged off (not incl. already charged off)
potential_loss_late = round(total_late_loan_amount - late_customers['total_payment'].sum())
potential_loss_late #£3,024,147


#loss of revenue of late and charged off customers 
potential_loss_late_charged_off = potential_loss_late + total_loss_charged_off
potential_loss_late_charged_off #£40,845,811.46


#creating a sebset to show data for customers who have already defaulted
default = transformed_loan_payments[transformed_loan_payments['loan_status'] == 'Default']

potential_revenue_default = round(default['loan_amount_incl_int_rate'].sum(), 2)
potential_revenue_default #£781,672.94

loss_from_default = round((default['loan_amount_incl_int_rate'] - default['total_payment']).sum(), 2)
loss_from_default #333,486.22

# % of overall revenue that late/charged off and default customers present (total value of loans)
overall_revenue_proportion = round((potential_revenue_charged_off + total_late_loan_amount + potential_revenue_default) / transformed_loan_payments['loan_amount_incl_int_rate'].sum() * 100, 2)   
overall_revenue_proportion #13.32%    
                      
# % of overall revenue that late/charged off and default customers present (potential loss as % of total revenue)
loss_choff_late_default = (loss_from_default + potential_loss_late_charged_off) / transformed_loan_payments['loan_amount_incl_int_rate'].sum() * 100
loss_choff_late_default #6.02%


 

#finding the unique values in loan status
unique_loan_status = transformed_loan_payments['loan_status'].unique()
unique_loan_status
"""['Current', 'Fully Paid', 'Charged Off', 'Late (31-120 days)',
       'In Grace Period', 'Late (16-30 days)', 'Default',
       'Does not meet the credit policy. Status:Fully Paid',
       'Does not meet the credit policy. Status:Charged Off'],
      dtype=object)"""

#counting number of customers late on their payments. I have included those who are in their Grace Period here as they've still missed a payment
late_customers = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Late | Grace')]

number_late_customers = len(late_customers)
number_late_customers #951


#calculating what percentage of customers are late on their payments
proportion_late = round(number_late_customers / len(transformed_loan_payments['loan_status']) * 100, 2)
proportion_late #1.75%


#total amount the 'late' loans are worth
total_late_loan_amount = round(late_customers['loan_amount_incl_int_rate'].sum(), 2)
total_late_loan_amount #£13,843,716.83


#loss to revenue if late --> charged off (not incl. already charged off)
potential_loss_late = round(total_late_loan_amount - late_customers['total_payment'].sum())
potential_loss_late #£3,024,147


#loss of revenue of late and charged off customers 
potential_loss_late_charged_off = potential_loss_late + total_loss_charged_off
potential_loss_late_charged_off #£40,845,811.46


#creating a sebset to show data for customers who have already defaulted
default = transformed_loan_payments[transformed_loan_payments['loan_status'] == 'Default']

potential_revenue_default = round(default['loan_amount_incl_int_rate'].sum(), 2)
potential_revenue_default #£781,672.94

loss_from_default = round((default['loan_amount_incl_int_rate'] - default['total_payment']).sum(), 2)
loss_from_default #333,486.22

# % of overall revenue that late/charged off and default customers present (total value of loans)
overall_revenue_proportion = round((potential_revenue_charged_off + total_late_loan_amount + potential_revenue_default) / transformed_loan_payments['loan_amount_incl_int_rate'].sum() * 100, 2)   
overall_revenue_proportion #13.32%    
                      
# % of overall revenue that late/charged off and default customers present (potential loss as % of total revenue)
loss_choff_late_default = (loss_from_default + potential_loss_late_charged_off) / transformed_loan_payments['loan_amount_incl_int_rate'].sum() * 100
loss_choff_late_default #6.02%



# ------------------------ Indicators for all lapsed ------------------- #

#subset
dfs_for_indicator_subset = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Late | Grace | Charged Off | Default')]
indicator_subset = dfs_for_indicator_subset


#changing str value columns to number codes
indicator_subset['loan_status'] = pd.Categorical(indicator_subset['loan_status']).codes

indicator_subset['home_ownership'] = pd.Categorical(indicator_subset['home_ownership']).codes

indicator_subset['grade'] = pd.Categorical(indicator_subset['grade']).codes

indicator_subset['last_payment_date'] = pd.Categorical(indicator_subset['last_payment_date']).codes

indicator_subset['next_payment_date'] = pd.Categorical(indicator_subset['next_payment_date']).codes

indicator_subset['earliest_credit_line'] = pd.Categorical(indicator_subset['earliest_credit_line']).codes

indicator_subset['employment_length'] = pd.Categorical(indicator_subset['employment_length']).codes

indicator_subset['verification_status'] = pd.Categorical(indicator_subset['verification_status']).codes

indicator_subset['issue_date'] = pd.Categorical(indicator_subset['issue_date']).codes

indicator_subset['purpose'] = pd.Categorical(indicator_subset['purpose']).codes



#chosen variables to check 
corr_indicator_subset = indicator_subset[['loan_amount', 'term', 'int_rate', 'instalment', 'annual_inc', 'dti', 'loan_status', 'home_ownership', 'purpose', 'total_payment', 'last_payment_date', 'next_payment_date', 'earliest_credit_line', 'total_accounts', 'open_accounts', 'mths_since_last_delinq', 'delinq_2yrs']]

#correlation matrix
corr_matrix = corr_indicator_subset.corr(method= 'kendall')
print(corr_matrix)
#largest indicator is last payment date (0.59), followed by next payment date (0.15). The third biggest indicator was annual income, although barely a correlation (-0.04)

#creating heatmap to show likelihood of each chosen variable being an indicator of loss 
cmap = sns.diverging_palette(20, 220, n=200)
heatmap = sns.heatmap(corr_matrix, cmap=cmap, square= True)
heatmap.set_title( "Potential Indicators of Loss")


# ----------------------------------------- visualising indicators -------------------------------------- #

#last_payment_date
scatter_last_payment = indicator_subset[['loan_status', 'last_payment_date']]
sns.lmplot(x="loan_status", y="last_payment_date", data=scatter_last_payment, line_kws={'color': 'red'})

#next_payment_date
scatter_next_payment = indicator_subset[['loan_status', 'next_payment_date']]
sns.lmplot(x="loan_status", y="next_payment_date", data=scatter_next_payment, line_kws={'color': 'red'})

#annual_income - mainly showing the lack of correlation in comparison to the above two
scatter_annual_inc = indicator_subset[['loan_status', 'annual_inc']]
sns.lmplot(x="loan_status", y="annual_inc", data=scatter_annual_inc, line_kws={'color': 'green'})


# -------------------------------- Indicators when Already Charged Off -------------------------------- #

#subset
loan_status_co_indicator_subset = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Charged Off')]
co_indicator_subset = loan_status_co_indicator_subset

co_indicator_subset['loan_status'] = pd.Categorical(co_indicator_subset['loan_status']).codes

co_indicator_subset['home_ownership'] = pd.Categorical(co_indicator_subset['home_ownership']).codes

co_indicator_subset['grade'] = pd.Categorical(co_indicator_subset['grade']).codes

co_indicator_subset['last_payment_date'] = pd.Categorical(co_indicator_subset['last_payment_date']).codes

co_indicator_subset['next_payment_date'] = pd.Categorical(co_indicator_subset['next_payment_date']).codes

co_indicator_subset['earliest_credit_line'] = pd.Categorical(co_indicator_subset['earliest_credit_line']).codes

co_indicator_subset['employment_length'] = pd.Categorical(co_indicator_subset['employment_length']).codes

co_indicator_subset['verification_status'] = pd.Categorical(co_indicator_subset['verification_status']).codes

co_indicator_subset['issue_date'] = pd.Categorical(co_indicator_subset['issue_date']).codes

co_indicator_subset['purpose'] = pd.Categorical(co_indicator_subset['purpose']).codes


#chosen variables to check 
corr_co_indicator_subset = co_indicator_subset[['loan_amount', 'term', 'int_rate', 'instalment', 'annual_inc', 'dti', 'loan_status', 'home_ownership', 'purpose', 'total_payment', 'last_payment_date', 'next_payment_date', 'earliest_credit_line', 'total_accounts', 'open_accounts', 'mths_since_last_delinq', 'delinq_2yrs']]

#correlation matrix
co_corr_matrix = corr_co_indicator_subset.corr(method= 'kendall')
print(co_corr_matrix)
#by far the largest indicator is the next payment date (0.98). After that it's the term length (-0.12). None of the other variables are of particular note. 
#This makes sense as being charged off is based around ability to make payments

#creating heatmap to show likelihood of each chosen variable being an indicator of loss 
cmap = sns.diverging_palette(20, 220, n=200)
heatmap = sns.heatmap(co_corr_matrix, cmap=cmap, square= True)
heatmap.set_title( "Indicators of Loss - Already Charged Off")

# ----------------------------------------- visualising indicators -------------------------------------- #

#next_payment_date
scatter_next_payment = co_indicator_subset[['loan_status', 'next_payment_date']]
sns.lmplot(x="loan_status", y="next_payment_date", data=scatter_next_payment, line_kws={'color': 'green'})

#last_payment_date
scatter_term = indicator_subset[['loan_status', 'term']]
sns.lmplot(x="loan_status", y="term", data=scatter_term, line_kws={'color': 'green'})

# ------------------------------------------ Potentially Charged Off ------------------------------------- #

#subset
loan_status_late_indicator_subset = transformed_loan_payments[transformed_loan_payments['loan_status'].str.contains('Late | Grace')]
late_indicator_subset = loan_status_late_indicator_subset

late_indicator_subset['loan_status'] = pd.Categorical(late_indicator_subset['loan_status']).codes

late_indicator_subset['home_ownership'] = pd.Categorical(late_indicator_subset['home_ownership']).codes

late_indicator_subset['grade'] = pd.Categorical(late_indicator_subset['grade']).codes

late_indicator_subset['last_payment_date'] = pd.Categorical(late_indicator_subset['last_payment_date']).codes

late_indicator_subset['next_payment_date'] = pd.Categorical(late_indicator_subset['next_payment_date']).codes

late_indicator_subset['earliest_credit_line'] = pd.Categorical(late_indicator_subset['earliest_credit_line']).codes

late_indicator_subset['employment_length'] = pd.Categorical(late_indicator_subset['employment_length']).codes

late_indicator_subset['verification_status'] = pd.Categorical(late_indicator_subset['verification_status']).codes

late_indicator_subset['issue_date'] = pd.Categorical(late_indicator_subset['issue_date']).codes

late_indicator_subset['purpose'] = pd.Categorical(late_indicator_subset['purpose']).codes


#chosen variables to check 
corr_late_indicator_subset = late_indicator_subset[['loan_amount', 'term', 'int_rate', 'instalment', 'annual_inc', 'dti', 'loan_status', 'home_ownership', 'purpose', 'total_payment', 'last_payment_date', 'next_payment_date', 'earliest_credit_line', 'total_accounts', 'open_accounts', 'mths_since_last_delinq', 'delinq_2yrs']]

#correlation matrix
late_corr_matrix = corr_late_indicator_subset.corr(method= 'kendall')
print(late_corr_matrix)
#The last payment date is the strongest indicator at 0.60. The next most correlated variable is the next payment date, 0.15. The third most, and last of note is total payment at -0.1. 
#It appears that the most likely problem that could lead to these customers becoming charged off is the weight of the payment schedule.

#creating heatmap to show likelihood of each chosen variable being an indicator of loss 
cmap = sns.diverging_palette(20, 220, n=200)
heatmap = sns.heatmap(late_corr_matrix, cmap=cmap, square= True)
heatmap.set_title( "Indicators of Loss - Potentially Charged Off")


# ----------------------------------------- visualising indicators --------------------------------------------- #

#last_payment_date
scatter_last_payment = late_indicator_subset[['loan_status', 'last_payment_date']]
sns.lmplot(x="loan_status", y="last_payment_date", data=scatter_last_payment, line_kws={'color': 'purple'})

#next_payment_date
scatter_next_payment = late_indicator_subset[['loan_status', 'next_payment_date']]
sns.lmplot(x="loan_status", y="next_payment_date", data=scatter_next_payment, line_kws={'color': 'purple'})

#total_payment
scatter_total_payment = late_indicator_subset[['loan_status', 'total_payment']]
sns.lmplot(x="loan_status", y="total_payment", data=scatter_total_payment, line_kws={'color': 'purple'})


