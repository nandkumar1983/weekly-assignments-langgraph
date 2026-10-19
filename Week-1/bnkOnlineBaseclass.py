
from bnkOnlineValidation import BankAccount,calculate_withdrawal_percentage



if __name__ == "__main__":

    try:
        account_balance = float(input("Enter the account balance: "))
        withdrawal_amount = float(input("Enter the withdrawal amount: "))
    except ValueError as error:
        raise ValueError("Invalid input: Please enter numeric values for balance and withdrawal amount.") from error

    account = BankAccount("123456789", account_balance, withdrawal_amount)

    withdrawal_percentage = calculate_withdrawal_percentage(account)
    if withdrawal_percentage is not None:
        print(f"Withdrawal percentage of balance: {withdrawal_percentage:.2f}%")
    