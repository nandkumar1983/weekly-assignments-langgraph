# Create the set of numbers from 1 to 20 with duplicate entry and print the set
numbers = {1, 2, 3, 4, 5, 6, 6, 8, 9, 10,12, 12, 13, 14, 15, 16, 17, 18, 19, 20, 10, 15} 
print(numbers)

#Add the new number as per user input and print the set check if number is alredy exits and print error message 
new_number = int(input("Enter a new number to add to the set: "))   
if new_number in numbers:
    print(f"{new_number} is already in the set.")
else:
    numbers.add(new_number)
    print(numbers)  

#Remove the number as per user input and print the set check if number is alredy exits and print error message
remove_number = int(input("Enter a number to remove from the set: "))
if remove_number in numbers:
    numbers.remove(remove_number)
    print(f"{remove_number} has been removed from the set.")
else:
    print(f"{remove_number} is not in the set.")
print(numbers)

#Check if any specfic number using membership operator and print the message
check_number = int(input("Enter a number to check if it exists in the set: "))
print(check_number in numbers) 
