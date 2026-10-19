#Create a class to get the employee attendance details and add the record into the file
class EmployeeAttendance:
    def __init__(self, emp_id, emp_name, attendance):
        self.emp_id = emp_id
        self.emp_name = emp_name
        self.attendance = attendance

#Validate attendance records file is present or not. If not create a new file and add the record into the file.
    def save_to_file(self):
        try:
            with open('attendance_records.txt', 'a') as file:
                file.write(f"{self.emp_id},{self.emp_name},{self.attendance}\n")
                print("Attendance record saved successfully.")
        except FileNotFoundError:
            with open('attendance_records.txt', 'w') as file:
                file.write(f"{self.emp_id},{self.emp_name},{self.attendance}\n")
                print("Attendance record saved successfully.")

#Create a method to read the attendance records from the file and display them.
    @staticmethod
    def read_from_file():
        try:
            with open('attendance_records.txt', 'r') as file:
                print("Employee Attendance Records:")
                for line in file:
                    emp_id, emp_name, attendance = line.strip().split(',')
                    print(f"ID: {emp_id}, Name: {emp_name}, Attendance: {attendance}")
        except FileNotFoundError:
            print("Attendance records file not found.")          