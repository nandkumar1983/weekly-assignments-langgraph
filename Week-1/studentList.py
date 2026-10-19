#Prepare the student list with names
students = ["Nandkumar", "Priya", "Aishwini", "Avisha"]
print(students)

#Add a new student to the list
students.append("Reyansh")
print(students) 

#Remove one student from the list as per the user input
student_to_remove = input("Enter the name of the student to remove: ")
if student_to_remove in students:
    students.remove(student_to_remove)
    print(f"{student_to_remove} has been removed from the list.")
else:
    print(f"{student_to_remove} is not in the list.")

print(students)

#Sort the student list in ascending order       
students.sort()
print("Sorted student list:", students) 