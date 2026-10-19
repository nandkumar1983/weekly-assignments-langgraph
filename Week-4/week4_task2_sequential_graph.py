import os
from pathlib import Path
from typing import TypedDict

from langchain_core.runnables.graph_mermaid import (
    draw_mermaid_png
)
from langgraph.graph import StateGraph, START, END


# ============================================================
# ASCEND PYTHON + GENAI + AGENTIC AI
# WEEK 4 - TASK 2
# USER PROFILE PROCESSING WORKFLOW
# OBJECTIVE: SEQUENTIAL GRAPH
# ============================================================


# ============================================================
# 1. FILE PATH CONFIGURATION
# ============================================================

CURRENT_DIRECTORY = Path(
    __file__
).resolve().parent

RUNTIME_GRAPH_MERMAID_FILE = (
    CURRENT_DIRECTORY
    / "task2_runtime_graph.mmd"
)

RUNTIME_GRAPH_PNG_FILE = (
    CURRENT_DIRECTORY
    / "task2_runtime_graph.png"
)


# ============================================================
# 2. DEFINE SHARED STATE
# ============================================================

class UserProfileState(TypedDict, total=False):
    """
    Shared state passed through all three sequential nodes.

    Input fields:
        name
        age
        skills

    Node output fields:
        greeting
        age_description
        skills_description

    Final output field:
        result
    """

    name: str
    age: int
    skills: list[str]

    greeting: str
    age_description: str
    skills_description: str

    result: str


# ============================================================
# 3. NODE 1 - GREETING GENERATOR
# ============================================================

def greeting_node(
    state: UserProfileState
) -> dict:
    """
    Generate a greeting using the user's name.

    This node updates only:
        greeting
    """

    name = state["name"].strip()

    if not name:
        raise ValueError(
            "Name cannot be empty."
        )

    greeting = (
        f"{name}, welcome to the system!"
    )

    print(
        "\n========================================"
    )

    print(
        "NODE 1 EXECUTED"
    )

    print(
        "GREETING GENERATOR"
    )

    print(
        "========================================"
    )

    print(
        "Received name:",
        name
    )

    print(
        "Generated greeting:",
        greeting
    )

    return {
        "greeting": greeting
    }


# ============================================================
# 4. NODE 2 - AGE DESCRIBER
# ============================================================

def age_node(
    state: UserProfileState
) -> dict:
    """
    Generate an age description.

    This node updates only:
        age_description
    """

    age = state["age"]

    if age < 0:
        raise ValueError(
            "Age cannot be negative."
        )

    age_description = (
        f"You are {age} years old!"
    )

    print(
        "\n========================================"
    )

    print(
        "NODE 2 EXECUTED"
    )

    print(
        "AGE DESCRIBER"
    )

    print(
        "========================================"
    )

    print(
        "Received age:",
        age
    )

    print(
        "Generated age description:",
        age_description
    )

    return {
        "age_description": age_description
    }


# ============================================================
# 5. FORMAT SKILLS
# ============================================================

def format_skills(
    skills: list[str]
) -> str:
    """
    Format the skills list as natural-language text.

    One skill:
        Python

    Two skills:
        Python and LangGraph

    Three skills:
        Python, Machine Learning, and LangGraph
    """

    cleaned_skills = [
        skill.strip()
        for skill in skills
        if skill.strip()
    ]

    if not cleaned_skills:
        raise ValueError(
            "At least one valid skill is required."
        )

    if len(cleaned_skills) == 1:
        return cleaned_skills[0]

    if len(cleaned_skills) == 2:
        return (
            f"{cleaned_skills[0]} "
            f"and {cleaned_skills[1]}"
        )

    return (
        ", ".join(cleaned_skills[:-1])
        + f", and {cleaned_skills[-1]}"
    )


# ============================================================
# 6. NODE 3 - SKILLS FORMATTER AND FINAL AGGREGATOR
# ============================================================

def skills_node(
    state: UserProfileState
) -> dict:
    """
    Format the user's skills and create the final result.

    This node updates:
        skills_description
        result
    """

    skills = state["skills"]

    formatted_skills = format_skills(
        skills
    )

    skills_description = (
        f"You have skills in: {formatted_skills}"
    )

    result = (
        f"{state['greeting']} "
        f"{state['age_description']} "
        f"{skills_description}"
    )

    print(
        "\n========================================"
    )

    print(
        "NODE 3 EXECUTED"
    )

    print(
        "SKILLS FORMATTER"
    )

    print(
        "========================================"
    )

    print(
        "Received skills:",
        skills
    )

    print(
        "Generated skills description:",
        skills_description
    )

    print(
        "Final aggregated result:",
        result
    )

    return {
        "skills_description":
            skills_description,

        "result":
            result
    }


# ============================================================
# 7. CREATE LANGGRAPH STATEGRAPH
# ============================================================

graph_builder = StateGraph(
    UserProfileState
)


# ============================================================
# 8. ADD THREE SEQUENTIAL NODES
# ============================================================

graph_builder.add_node(
    "greeting_node",
    greeting_node
)

graph_builder.add_node(
    "age_node",
    age_node
)

graph_builder.add_node(
    "skills_node",
    skills_node
)


# ============================================================
# 9. DEFINE SEQUENTIAL EDGES
#
# START
#   |
#   v
# greeting_node
#   |
#   v
# age_node
#   |
#   v
# skills_node
#   |
#   v
# END
# ============================================================

graph_builder.add_edge(
    START,
    "greeting_node"
)

graph_builder.add_edge(
    "greeting_node",
    "age_node"
)

graph_builder.add_edge(
    "age_node",
    "skills_node"
)

graph_builder.add_edge(
    "skills_node",
    END
)


# ============================================================
# 10. COMPILE GRAPH
# ============================================================

profile_graph = graph_builder.compile()


# ============================================================
# 11. PARSE TERMINAL SKILLS
# ============================================================

def parse_skills(
    skills_input: str
) -> list[str]:
    """
    Convert comma-separated terminal text into a list.

    Example:

        Python, Machine Learning, LangGraph
    """

    skills = [
        skill.strip()
        for skill in skills_input.split(",")
        if skill.strip()
    ]

    if not skills:
        raise ValueError(
            "Enter at least one skill."
        )

    return skills


# ============================================================
# 12. MERMAID TEXT ESCAPING
# ============================================================

def escape_mermaid_text(
    value: object
) -> str:
    """
    Escape text before inserting it into Mermaid labels.
    """

    text_value = str(value)

    return (
        text_value
        .replace("&", "&amp;")
        .replace('"', "'")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", " ")
    )


# ============================================================
# 13. BUILD RUNTIME-SPECIFIC MERMAID GRAPH
# ============================================================

def build_runtime_mermaid(
    input_state: UserProfileState,
    output_state: UserProfileState
) -> str:
    """
    Build a runtime graph containing the actual inputs and
    the output generated by each node.

    This graph is generated only after profile_graph.invoke().
    """

    name = escape_mermaid_text(
        input_state["name"]
    )

    age = input_state["age"]

    skills = escape_mermaid_text(
        ", ".join(input_state["skills"])
    )

    greeting = escape_mermaid_text(
        output_state["greeting"]
    )

    age_description = escape_mermaid_text(
        output_state["age_description"]
    )

    skills_description = escape_mermaid_text(
        output_state["skills_description"]
    )

    result = escape_mermaid_text(
        output_state["result"]
    )

    runtime_mermaid = f"""
---
config:
  flowchart:
    curve: linear
---
flowchart TD

    START(["START"])

    GREETING["<b>Node 1: Greeting Generator</b><br/><br/>Input Name: {name}<br/><br/>Output:<br/>{greeting}"]

    AGE["<b>Node 2: Age Describer</b><br/><br/>Input Age: {age}<br/><br/>Output:<br/>{age_description}"]

    SKILLS["<b>Node 3: Skills Formatter</b><br/><br/>Input Skills: {skills}<br/><br/>Output:<br/>{skills_description}<br/><br/><b>Final Result:</b><br/>{result}"]

    END(["END"])

    START --> GREETING
    GREETING --> AGE
    AGE --> SKILLS
    SKILLS --> END

    classDef startNode fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20;
    classDef greetingNode fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1;
    classDef ageNode fill:#FFF8E1,stroke:#F9A825,stroke-width:2px,color:#E65100;
    classDef skillsNode fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C;
    classDef endNode fill:#ECEFF1,stroke:#455A64,stroke-width:2px,color:#263238;

    class START startNode;
    class GREETING greetingNode;
    class AGE ageNode;
    class SKILLS skillsNode;
    class END endNode;
""".strip()

    return runtime_mermaid


# ============================================================
# 14. GENERATE RUNTIME GRAPH
# ============================================================

def generate_runtime_graph(
    input_state: UserProfileState,
    output_state: UserProfileState
) -> None:
    """
    Generate and open the runtime graph after the complete
    sequential workflow finishes.
    """

    print(
        "\n========================================"
    )

    print(
        "GENERATING RUNTIME SEQUENTIAL GRAPH"
    )

    print(
        "========================================"
    )

    print(
        "\nExecution path:"
    )

    print(
        "START -> greeting_node -> age_node "
        "-> skills_node -> END"
    )

    runtime_mermaid = build_runtime_mermaid(
        input_state=input_state,
        output_state=output_state
    )

    # --------------------------------------------------------
    # Save runtime Mermaid
    # --------------------------------------------------------

    RUNTIME_GRAPH_MERMAID_FILE.write_text(
        runtime_mermaid,
        encoding="utf-8"
    )

    print(
        "\nRuntime Mermaid graph saved:"
    )

    print(
        RUNTIME_GRAPH_MERMAID_FILE
    )

    # --------------------------------------------------------
    # Generate PNG
    # --------------------------------------------------------

    try:
        png_data = draw_mermaid_png(
            mermaid_syntax=runtime_mermaid,
            background_color="white",
            padding=20,
            max_retries=2,
            retry_delay=1.0
        )

        RUNTIME_GRAPH_PNG_FILE.write_bytes(
            png_data
        )

        print(
            "\nRuntime graph PNG generated:"
        )

        print(
            RUNTIME_GRAPH_PNG_FILE
        )

        # ----------------------------------------------------
        # Open automatically on Windows
        # ----------------------------------------------------

        if os.name == "nt":
            try:
                os.startfile(
                    str(RUNTIME_GRAPH_PNG_FILE)
                )

                print(
                    "\nRuntime graph opened automatically."
                )

            except OSError as open_error:
                print(
                    "\nThe graph was generated but Windows "
                    "could not open it automatically."
                )

                print(
                    "Open error:",
                    open_error
                )

    except Exception as graph_error:
        print(
            "\nRuntime graph PNG generation failed."
        )

        print(
            "Error type:",
            type(graph_error).__name__
        )

        print(
            "Error details:",
            graph_error
        )

        print(
            "\nThe Mermaid runtime graph was still saved."
        )


# ============================================================
# 15. EXECUTE USER PROFILE WORKFLOW
# ============================================================

def execute_profile_workflow(
    name: str,
    age: int,
    skills: list[str]
) -> UserProfileState:
    """
    Execute the workflow in this order:

        Terminal Input
            ↓
        Input State
            ↓
        greeting_node
            ↓
        age_node
            ↓
        skills_node
            ↓
        Final Result
            ↓
        Generate Runtime Graph
    """

    input_state: UserProfileState = {
        "name": name,
        "age": age,
        "skills": skills
    }

    # --------------------------------------------------------
    # Display input state
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "LANGGRAPH INPUT STATE"
    )

    print(
        "========================================"
    )

    print("{")

    print(
        f'  "name": "{name}",'
    )

    print(
        f'  "age": {age},'
    )

    print(
        f'  "skills": {skills}'
    )

    print("}")

    # --------------------------------------------------------
    # Invoke actual LangGraph
    # --------------------------------------------------------

    print(
        "\nExecuting sequential LangGraph..."
    )

    print(
        "START -> greeting_node -> age_node "
        "-> skills_node -> END"
    )

    final_state = profile_graph.invoke(
        input_state
    )

    # --------------------------------------------------------
    # Display complete final state
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "LANGGRAPH FINAL STATE"
    )

    print(
        "========================================"
    )

    print(
        "Greeting:"
    )

    print(
        final_state["greeting"]
    )

    print(
        "\nAge Description:"
    )

    print(
        final_state["age_description"]
    )

    print(
        "\nSkills Description:"
    )

    print(
        final_state["skills_description"]
    )

    # --------------------------------------------------------
    # Display required final output
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "FINAL OUTPUT"
    )

    print(
        "========================================"
    )

    print("{")

    print(
        f'  "result": "{final_state["result"]}"'
    )

    print("}")

    # --------------------------------------------------------
    # Generate graph only after LangGraph execution
    # --------------------------------------------------------

    generate_runtime_graph(
        input_state=input_state,
        output_state=final_state
    )

    return final_state


# ============================================================
# 16. MAIN APPLICATION
# ============================================================

def main() -> None:
    """
    Run Week 4 Task 2.

    The graph is not generated during startup.

    The runtime graph is generated after the sequential
    workflow successfully completes.
    """

    print(
        "\n========================================"
    )

    print(
        "ASCEND PYTHON LANGGRAPH"
    )

    print(
        "WEEK 4 - TASK 2"
    )

    print(
        "USER PROFILE PROCESSING WORKFLOW"
    )

    print(
        "OBJECTIVE: SEQUENTIAL GRAPH"
    )

    print(
        "========================================"
    )

    print(
        "\nThe graph will be generated after the "
        "profile workflow executes."
    )

    # --------------------------------------------------------
    # Name
    # --------------------------------------------------------

    name = input(
        "\nEnter name: "
    ).strip()

    if not name:
        raise ValueError(
            "Name cannot be empty."
        )

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    age_input = input(
        "Enter age: "
    ).strip()

    try:
        age = int(age_input)

    except ValueError as age_error:
        raise ValueError(
            "Age must be a valid whole number."
        ) from age_error

    if age < 0:
        raise ValueError(
            "Age cannot be negative."
        )

    # --------------------------------------------------------
    # Skills
    # --------------------------------------------------------

    print(
        "\nEnter skills separated by commas."
    )

    print(
        "Example:"
    )

    print(
        "Python, Machine Learning, LangGraph"
    )

    skills_input = input(
        "\nEnter skills: "
    ).strip()

    skills = parse_skills(
        skills_input
    )

    # --------------------------------------------------------
    # Execute the sequential LangGraph
    # --------------------------------------------------------

    execute_profile_workflow(
        name=name,
        age=age,
        skills=skills
    )


# ============================================================
# 17. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\n\nTask 2 stopped by user."
        )

    except Exception as application_error:
        print(
            "\n========================================"
        )

        print(
            "TASK 2 FAILED"
        )

        print(
            "========================================"
        )

        print(
            "Error type:",
            type(application_error).__name__
        )

        print(
            "Error details:",
            application_error
        )

        print(
            "\nTask 2 terminated."
        )   