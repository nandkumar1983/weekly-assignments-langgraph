#Create the tuples with 5 subject names and print the tuple

subjects=("Maths","Science","English","Hindi","Marathi")
print(subjects) 

#   Print first and last subject name from the tuple 

for subject in subjects:
    print(subjects[0]) #First subject name
    print(subjects[-1]) #last subject name
    break  

# Print count of subjects in the tuple
print(len(subjects)) #count of subjects in the tuple

# Find the subject name  in the tuple and print the message if found or not found
userSubject=input("Enter the subject name to search: ")

if {userSubject} in subjects:
    print("Subject is found in the tuple")  
else:
    print("Subject is not found in the tuple")  