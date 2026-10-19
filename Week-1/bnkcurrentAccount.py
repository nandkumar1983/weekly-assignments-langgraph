#Derived Class: CurrentAccount
from bnkaccountsBaseClass import Account


class CurrentAccount(Account):  
    def __init__(self, account_number, account_holder, balance, overdraft_limit):
        super().__init__(account_number, account_holder, balance)
        self.overdraft_limit = overdraft_limit

    #Override withdraw method to allow overdraft
    def withdraw(self, amount):
        if self.balance + self.overdraft_limit >= amount:
            self.balance -= amount
        else:
            print("Insufficient funds and overdraft limit reached")