from datetime import datetime
from typing import List
import pandas as pd
import pickle
import os
from datascience.src.user_data.User import User as UserClass  # avoid conflict with User
from datascience.src.user_data.UserDataExtractor import UserDataExtractor

from fastapi import FastAPI
from pydantic import BaseModel, validator

naf_enc = pickle.load(open(os.path.join('datascience','pickle', 'naf_enc.pickle'), 'rb'))
job_type_enc = pickle.load(open(os.path.join('datascience','pickle', 'job_type_enc.pickle'), 'rb'))
last_letter_enc = pickle.load(open(os.path.join('datascience','pickle', 'last_letter_enc.pickle'), 'rb'))
model = pickle.load(open(os.path.join('datascience','pickle', 'lr_exp.pickle'), 'rb'))
features = pickle.load(open(os.path.join('datascience','pickle', 'features.pickle'), 'rb'))


class User(BaseModel):
    update_date: datetime
    business_NAF_code: str
    id: int


class Account(BaseModel):
    user_id: int
    balance: float
    id: int


class Transaction(BaseModel):
    account_id: int
    amount: float
    date: datetime


class RequestPredict(BaseModel):
    user: User
    accounts: List[Account]
    transactions: List[Transaction]

    @validator("transactions")
    def validate_transaction_history(cls, v, *, values):
        # validate that
        # - the transaction list passed has at least 6 months history
        # - no transaction is posterior to the user's update date
        if len(v) < 1:
            raise ValueError("Must have at least one Transaction")

        update_t = values["user"].update_date

        oldest_t = v[0].date
        newest_t = v[0].date
        for t in v[1:]:
            if t.date < oldest_t:
                oldest_t = t.date
            if t.date > newest_t:
                newest_t = t.date

        assert (
            update_t - newest_t
        ).days >= 0, "Update Date Inconsistent With Transaction Dates"
        assert (update_t - oldest_t).days > 183, "Not Enough Transaction History"

        return v


class ResponsePredict(BaseModel):
    user_id: int
    predicted_amount: float

# just for docs
class Data(BaseModel):
    income_3 : float
    expense_3: float
    agg_3 : float
    month_std_3 : float
    month_end_bal_3: float
    income_2 : float
    expense_2 : float
    agg_2: float
    month_std_2 : float
    month_end_bal_2: float
    income_1: float
    expense_1 : float
    agg_1 : float
    month_std_1: float
    month_end_bal_1: float
    last_month_gains : float
    month_predicted : int
    naf_code : int
    job_type: int
    last_letter_naf: int

def predict(
    transactions: List[Transaction], accounts: List[Account], user: User
) -> float:
    transactions_df = pd.DataFrame(map(dict, transactions))
    accounts_df = pd.DataFrame(map(dict, accounts))
    print(accounts_df['user_id'])
    current_user = UserClass(user.id, accounts_df, transactions_df,  user.business_NAF_code)
    data = preprocess_user_data(current_user)
    prediction = model.predict(data)

    return prediction

def preprocess_user_data(current_user:User) -> pd.DataFrame:
    current_user_data = UserDataExtractor(current_user)
    last_date = current_user_data.end_date
    past_data_df = current_user_data.construct_past_data(last_date, 3, current_user_data.filled_history)
    data = pd.concat([past_data_df, current_user.info], axis=1)
    # encode naf code features
    data['naf_code'] = naf_enc.transform(data['naf_code'])
    data['job_type'] = job_type_enc.transform(data['job_type'])
    data['last_letter_naf'] = last_letter_enc.transform(data['last_letter_naf'])
    return data


app = FastAPI()


@app.post("/predict")
async def root(predict_body: RequestPredict):
    transactions = predict_body.transactions
    accounts = predict_body.accounts
    user = predict_body.user

    predicted_amount = predict(transactions, accounts, user)[0]
    # Return predicted amount along with account id
    return {"user_id": user.id, "predicted_amount": predicted_amount}
