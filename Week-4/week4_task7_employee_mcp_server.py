import json

from fastmcp import FastMCP


# ============================================================
# ASCEND PYTHON LANGGRAPH
# WEEK 4 - TASK 7
# EMPLOYEE DIRECTORY MCP SERVER
# ============================================================


# ============================================================
# 1. CREATE MCP SERVER
# ============================================================

employee_mcp_server = FastMCP(
    "Employee Directory MCP Server"
)


# ============================================================
# 2. LOCAL EMPLOYEE DATA
#
# This data is exposed only through MCP tools.
# The LangGraph assistant must not access it directly.
# ============================================================

EMPLOYEES = {
    "Vaishali": {
        "designation": "Manager",
        "manager": "Rupali Giri",
    },
    "Rupali Giri": {
        "designation": "Senior Manager",
        "manager": "Samuel Manjula",
    },
}


# ============================================================
# 3. NORMALIZE EMPLOYEE NAME
# ============================================================

def find_employee_name(
    requested_name: str,
) -> str | None:
    """
    Find an employee using case-insensitive matching.
    """

    normalized_requested_name = (
        requested_name
        .strip()
        .lower()
    )

    for employee_name in EMPLOYEES:

        if (
            employee_name.lower()
            ==
            normalized_requested_name
        ):

            return employee_name

    return None


# ============================================================
# 4. MCP TOOL
# ============================================================

@employee_mcp_server.tool
def get_employee_details(
    employee_name: str,
) -> str:
    """
    Get employee-directory information for a named employee.

    Use this tool when the user asks for an employee's
    designation, manager, or complete directory details.

    Args:
        employee_name:
            Full or known employee name.

    Returns:
        JSON containing the employee name, designation,
        manager, and lookup status.
    """

    matched_name = find_employee_name(
        employee_name
    )

    if matched_name is None:

        result = {
            "status": "not_found",
            "requested_employee": employee_name,
            "message": (
                f"No employee record was found for "
                f"'{employee_name}'."
            ),
            "available_employees": list(
                EMPLOYEES.keys()
            ),
        }

        return json.dumps(
            result,
            indent=2,
        )

    employee_record = EMPLOYEES[
        matched_name
    ]

    result = {
        "status": "success",
        "employee_name": matched_name,
        "designation": employee_record[
            "designation"
        ],
        "manager": employee_record[
            "manager"
        ],
    }

    return json.dumps(
        result,
        indent=2,
    )


# ============================================================
# 5. OPTIONAL DIRECTORY-LISTING TOOL
# ============================================================

@employee_mcp_server.tool
def list_employee_names() -> str:
    """
    List the employee names available in the local directory.

    Use this tool only when the user asks which employees
    are available in the directory.
    """

    result = {
        "status": "success",
        "employees": list(
            EMPLOYEES.keys()
        ),
    }

    return json.dumps(
        result,
        indent=2,
    )


# ============================================================
# 6. STANDALONE SERVER ENTRY POINT
#
# The Task 7 client uses this MCP server in-process, so
# running this file separately is optional.
# ============================================================

if __name__ == "__main__":

    employee_mcp_server.run()