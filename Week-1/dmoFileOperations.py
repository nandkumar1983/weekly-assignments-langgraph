#Create Directories and File Using Python pathlib

import pathlib

#Create directory structure data/reports/2026
path = pathlib.Path("data/reports/2026")
path.mkdir(parents=True, exist_ok=True)

#Create a file named report.txt in the 2026 directory
file_path = path / "Monthly_Report.txt"
file_path.touch(exist_ok=True)  

#Write some content to the file
file_path.write_text("This is the monthly report for 2026.\n")
with file_path.open("a") as report_file:
	report_file.write("File operation demo using pathlib module in python.\n")

# Display directory path, file path ,absolute path and file content
print(f"Directory Path: {path}")
print(f"File Path: {file_path}")
print(f"Absolute Path: {file_path.resolve()}")
print(f"File Content:\n{file_path.read_text()}")    



