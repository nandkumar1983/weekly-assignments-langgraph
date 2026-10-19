#Day1- Task5 : Python Dictionaries -Learn to store and access data using key-value pairs.
studentinfo = {}
studentinfo["Name"] = "Nandkumar Jadhav"
studentinfo["Roll Number"] = 27
studentinfo["Course"] = "Python"


print("Student Information:",studentinfo)

#Add new key-value pair to the dictionary
studentinfo["Marks"] = 89.99

print("Updated Student Information:",studentinfo)

#Update the course name in the dictionary
studentinfo["Course"] = "Agentic AI"

print("Student Information after Update:",studentinfo)

#Display all the keys and values seperately using for loop
print("Keys in the dictionary:")    
for key in studentinfo:
    print(key)
    
print("Values in the dictionary:")
for value in studentinfo.values():
    print(value)