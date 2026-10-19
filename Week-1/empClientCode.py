from empSalaryCalculation import FullTimeEmployee, PartTimeEmployee

if __name__ == "__main__":
    full_time_emp = FullTimeEmployee("Nandkumar Jadhav", 101, 5000)
    part_time_emp = PartTimeEmployee("Avisha Jadhav", 102, 200, 80)

    print(f"{full_time_emp.name}'s Salary: {full_time_emp.calculate_salary()}")
    print(f"{part_time_emp.name}'s Salary: {part_time_emp.calculate_salary()}")