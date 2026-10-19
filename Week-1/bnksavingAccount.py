#Derived Class: SavingsAccount
from bnkaccountsBaseClass import Account


class SavingsAccount(Account):
    def __init__(self, account_number, account_holder, balance, interest_rate):
        super().__init__(account_number, account_holder, balance)
        self.interest_rate = interest_rate

    #Method to calculate interest 
    def calculate_interest(self):
        interest = self.balance * (self.interest_rate / 100)
        self.balance += interest
        return interest, self.interest_rate