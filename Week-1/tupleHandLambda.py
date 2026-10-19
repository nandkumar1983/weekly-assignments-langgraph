tup_list=[(1,4),(2,3),(3,8),(4,9),(5,87)]

#Using lambda function to sort the list of tuples based on the second element of each tuple 
tup_list.sort(key=lambda x: x[1])

print("Sorted list of tuples based on the second element:")
for tup in tup_list:    
    print(tup)  

# Create a list of dictionaries with each person's name and age
familyMembers = [
    {"Name": "Nandkumar Jadhav", "Age": 42},
    {"Name": "Reyansh Jadhav", "Age": 5},
    {"Name": "Avisha Jdhav", "Age": 2}
]

# Using a lambda function to sort the people based on age
sorted_people = sorted(familyMembers, key=lambda person: person["Age"])

# Print the sorted people
print("Sorted people based on age:")
for person in sorted_people:
    print(f"{person['Name']}: {person['Age']}")
