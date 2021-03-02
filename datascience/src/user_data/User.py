import pandas as pd
from datascience.src.user_data.utils import get_job_type

"""
Encapsulates data related to a user, reconstructs user history"""
class User:

    def __init__(self, user_id, accounts, transactions, naf_code):
        self.id = user_id
        self.naf_code = naf_code
        self._add_naf_code_features()
        self._build_user_history(accounts, transactions)

    def _build_user_history(self, accounts, transactions):
        self.user_accounts = self._fetch_user_accounts(accounts)
        self.account_ids = list(self.user_accounts['id'])
        self.account_history = self._fetch_transaction_history(transactions)
        last_balance_sum = self._fetch_last_balance()
        self._add_balance_histories(last_balance_sum)

    # fetch ids of all the accounts under this user
    def _fetch_user_accounts(self, accounts):
        return accounts.loc[accounts['user_id'] == self.id]

    # fetch the total combined balance of all accounts at the time of update
    def _fetch_last_balance(self):
        account_balances = []
        for i, account in self.user_accounts.iterrows():
            account_balances.append(account['balance'])
        return sum(account_balances)


    def _fetch_transaction_history(self, transactions):
        account_histories = []
        for account_id in self.account_ids:
            account_histories.append \
                (transactions.loc[transactions['account_id'] == account_id].drop(['account_id'], axis=1))
        return pd.concat(account_histories, axis=0)

    """
    Input: balance at last update
    Output: None
    Does: Adds 'balance' column to all account histories, representing the balance just before a transaction
    """
    def _add_balance_histories(self, last_account_balance):
        balance_hist = []
        bal = last_account_balance
        for _, tr in self.account_history[::-1].iterrows():
            bal += -tr['amount']
            balance_hist.append(round(bal, 3))

        self.account_history['balance'] = list(reversed(balance_hist))

    # get number of days of account history
    def get_history_length(self):
        d = (self.account_history['date'].values[-1] - self.account_history['date'].values[0]).astype('timedelta64[D]')
        return int(str(d).split()[0])

    # extracts features from naf code to later be encoded
    def _add_naf_code_features(self):
        self.info = pd.DataFrame()
        self.info['naf_code'] = [self.naf_code]
        # naf code job type (first word of naf code name)
        self.info['job_type'] = [get_job_type(self.naf_code)]
        # business NAF code category (last letter of code)
        self.info['last_letter_naf'] = [self.naf_code[-1]]
        # avoids having index in the features
        self.info.reset_index(drop=True)

