from ast import alias

from pydantic import BaseModel, Field
from stdDBOperations import StudentDBOperations

# Create the student class

class Studentinfo(BaseModel):
	student_id: int = Field(...,description="Unique Student ID",alias="student_id")
	student_name:str = Field(...,description="Student name",alias="student_name")
	student_class:str=Field(...,description="Student Class",alias="student_class")
	student_age:int=Field(...,description="Student Age",alias="student_age")
	
# Check unique student id by using hash method

def __hash__(self):
	return hash(self.student_id)

def __eq__(self, other):
		if isinstance(other, Studentinfo):
			return self.student_id == other.student_id
			return False


# Call the StudentDBOperations class to fetch database records into Student Object
std_db_operations = StudentDBOperations()