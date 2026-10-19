#Title: Online Banking Withdrawal System
# bank customer wants to withdraw money from their account. The program should handle invalid inputs and insufficient balance situation

class BankAccount:
    def __init__(self, account_number, balance,withdrwal_balance):
        self.balance = balance
        self.withdrawal_balance = withdrwal_balance
        self.account_number = account_number

#Raise a custom exception InsufficientBalanceError if withdrawal amount is greater than available balance.
class InsufficientBalanceError(Exception):
    pass


#Raise a ZeroDivisionError by calculating withdrawal percentage of balance.

def calculate_withdrawal_percentage(account):
    try:
        withdrawal_percentage = (account.withdrawal_balance / account.balance) * 100
        if withdrawal_percentage > 100:
            raise InsufficientBalanceError(f"Insufficient balance for the withdrawal. {withdrawal_percentage:.2f}% of balance requested.")
        return withdrawal_percentage
    except ZeroDivisionError:
        print("Error: Balance is zero, cannot calculate withdrawal percentage.")
        return None

