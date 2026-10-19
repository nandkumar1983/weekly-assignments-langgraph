from bnkaccountsBaseClass import Account
from bnksavingAccount import SavingsAccount
from bnkcurrentAccount  import CurrentAccount

# Create a SavingsAccount object and a CurrentAccount object
savings_account = SavingsAccount("SA123", "Nandkumar Jadhav", 1000, 9.5)
current_account = CurrentAccount("CA456", "Reyansh Jadhav", 2000, 500)

# Display account information for both accounts
print("Savings Account Information:")   
savings_account.display_account_info()
print("\nCurrent Account Information:")
current_account.display_account_info()  

# Perform deposit and withdraw operations on both accounts
savings_account.deposit(1500)
current_account.deposit(1000)

# Display updated account information after deposit
print("\nUpdated Savings Account Information after Deposit:")
savings_account.display_account_info()
print("\nUpdated Current Account Information after Deposit:")
current_account.display_account_info()

# Withdraw funds from both accounts
savings_account.withdraw(500)
current_account.withdraw(200)  

# Display updated account information after withdrawal
print("\nUpdated Savings Account Information after Withdrawal:")        
savings_account.display_account_info()
print("\nUpdated Current Account Information after Withdrawal:")
current_account.display_account_info()

# Calculate and display interest for the savings account
print("\nUpdated Savings Account Information after Interest Application:")
interest, rate = savings_account.calculate_interest()
savings_account.display_account_info()
print(f"Interest Earned: ${interest:.2f} at {rate}% interest rate")

#Demonstrate overdraft withdrawal in the current account
print("\nOverdraft Withdrawal from Current Account:")
#Overdraft limit reached
current_account.withdraw(5500) 
#Within Overdraft limit
current_account.withdraw(1500) 
current_account.display_account_info()  