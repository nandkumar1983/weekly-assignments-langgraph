def grade_calculator(score) -> str:
    if score >= 90:
        return "A"
    elif score >= 75 and score < 90:
        return "B"
    elif score > 60 and score < 75:
        return "C"
    elif score >= 50 and score <= 60:
        return "D"
    else:
        return "Fail"

print("Student Grade:", grade_calculator(85))  # Output: B
print("Student Grade:", grade_calculator(95))  # Output: A
print("Student Grade:", grade_calculator(65))  # Output: C
print("Student Grade:", grade_calculator(55))  # Output: D
print("Student Grade:", grade_calculator(45))  # Output: Fail