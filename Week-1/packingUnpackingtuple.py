# Demonstrate packing and unpacking with a tuple, list, and dictionary.

student_details = (101, "Nandkumar Jadhav", "A")
subject_marks = [85, 90, 88]
additional_info = {"age": 42, "course": "Python"}
city_info = {"city": "Pune"}

roll_no, name, grade = student_details
english, mathematics, science = subject_marks
student_info = {**additional_info, **city_info}

print("Student Details")
print(f"Roll No: {roll_no}")
print(f"Name: {name}")
print(f"Grade: {grade}")
print(f"English: {english}")
print(f"Mathematics: {mathematics}")
print(f"Science: {science}")
print(f"Age: {student_info['age']}")
print(f"Course: {student_info['course']}")
print(f"City: {student_info['city']}")
