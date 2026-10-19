def calculate_total(n: int) -> float:
    total = 0
    for i in range(1, n + 1):
        value = float(input(f"Enter the item {i} value: "))
        total += value
    return total

print("Enter the number of items to calculate total:")
n = int(input())
print("Total of ",n," items: ", calculate_total(n))
