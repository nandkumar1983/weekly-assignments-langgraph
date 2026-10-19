from empAttendance import EmployeeAttendance

# Get employee attendance details from user input
employee_id = input("Enter Employee ID: ")
employee_name = input("Enter Employee Name: ")  
employee_attendance = input("Enter Attendance (Present/Absent): ")

# Create an instance of EmployeeAttendance and save the record to the file
employee = EmployeeAttendance(employee_id, employee_name, employee_attendance)
employee.save_to_file()


print("Attendance records:")
employee.read_from_file()