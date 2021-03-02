import pandas as pd
import numpy as np

from datascience.src.user_data.utils import get_month

"""
Extracts data relevant to our prediction from a User object
"""
class UserDataExtractor:

    def __init__(self, user):
        self.user = user
        self.start_date = user.account_history['date'].values[0]
        self.end_date = user.account_history['date'].values[-1]
        self.filled_history = self._construct_filled_history()

    """
    Construct the even timeseries with 1 entry per day
    Aggregate daily transactions if there are more than 1
    Fill in with 0.0 when no transaction occurs
    """
    def _construct_filled_history(self):
        filled_history = []
        row = 0
        # from user start date to final update
        for day in np.arange(self.start_date, self.end_date+np.timedelta64(1,'D'), dtype='datetime64[D]'):
            day_amount = 0.0
            day_start_balance = self.user.account_history.iloc[row]['balance']

            # aggregate transactions made that day
            while row < len(self.user.account_history) and self.user.account_history.iloc[row]['date'] == day:
                day_amount += self.user.account_history.iloc[row]['amount']
                row += 1
            # add day to filled history, with amount 0 if no transactions occur
            filled_history.append({
                'date': day,
                'amount': day_amount,
                'balance': day_start_balance
            })
        return pd.DataFrame(filled_history).set_index('date')

    def construct_past_data(self, until_date, past_months, history):
        past_data = pd.DataFrame()

        # iterate through month_back months of history ending at until_date
        for month_back in range(past_months, 0, -1):
            month_start = until_date - np.timedelta64(month_back * 30, 'D')
            month_end = month_start + np.timedelta64(30, 'D')
            timeframe = history[month_start:month_end]

            income = self.calculate_income(timeframe)
            expense = self.calculate_expense(timeframe)
            minus_month = str(month_back)

            # extract features from account data for each month back
            past_data['income_'+minus_month] = [income]
            past_data['expense_'+minus_month] = [expense]
            past_data['agg_'+minus_month] = [income + expense]
            past_data['month_std_' + minus_month] = [timeframe['amount'].std()]
            past_data['month_end_bal_'+minus_month] = [history.loc[month_end]['balance']]

        # add last month aggregate
        past_data['last_month_gains'] = [income + expense]
        # add id of month where the prediction starts (to capture seasonality across the dataset)
        past_data['month_predicted'] = [get_month(until_date)]

        return past_data

    def calculate_income(self, timeframe):
        if len(timeframe) == 0:
            raise Exception("Not enough data for user ", self.user.id)
        income = timeframe['amount'].apply(lambda x: max(x, 0)).sum()
        return income

    def calculate_expense(self, timeframe):
        if len(timeframe) == 0:
            raise Exception("Not enough data for user ", self.user.id)
        expense = timeframe['amount'].apply(lambda x: min(x, 0)).sum()
        return expense
