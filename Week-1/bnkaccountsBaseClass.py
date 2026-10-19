#Create a Python application to manage different types of bank accounts using inheritance.
#Base Class: Account
class Account:
    def __init__(self, account_number, account_holder, balance):
        self.account_number = account_number
        self.account_holder = account_holder
        self.balance = balance

    #Deposit method to add funds to the account
    def deposit(self, amount):
        self.balance += amount

    #Withdraw method to remove funds from the account
    def withdraw(self, amount):
        if self.balance >= amount:
            self.balance -= amount
        else:
            print("Insufficient funds")

    #Display account information method
    def display_account_info(self):
        print(f"Account Number: {self.account_number}")
        print(f"Account Holder: {self.account_holder}")
        print(f"Balance: ${self.balance}")



