# Prepare the 10 to 100 number list as tuples
numList=(10,20,30,40,50,60,70,80,90,100)

#Print first 5 elements from the tuple
print(numList[0:5]) 

#Print elements from index 2 to 6
print(numList[2:7])

#Print elements from beginning to index 7
print(numList[:8])

#Print elements from index 4 to end of the tuple
print(numList[4:])

#Print every second element from the tuple
print(numList[::2])

#Print every third element from the tuple
print(numList[::3])

#Complete tuple list elements 
print(numList)  

# Print only last 3 elements from the tuple
print(numList[-3:])

# print last 5 elemts from the tuple
print(numList[-5:])

# print all elements except 2nd and 7th elements from the tuple
print(numList[0:1]+numList[2:6]+numList[7:10])

# print every 2nd elements from the last from the tuple
print(numList[::-2])

# Print tuples in reverse order
print(numList[::-1])

# Reverse last 6 elements from the tuple
print(numList[-1:-7:-1])

#elements from index -2 to beginning in reverse order in the tuple
print(numList[-2::-1])

#every 3rd elements in reverse order in the tuple   
print(numList[::-3])