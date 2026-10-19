import os
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.runnables.graph_mermaid import (
    draw_mermaid_png
)
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START, END


# ============================================================
# ASCEND PYTHON LANGGRAPH
# WEEK 4 - TASK 4
# IT HELPDESK TROUBLESHOOTING AGENT LOOP
# ============================================================


# ============================================================
# 1. FILE PATH CONFIGURATION
# ============================================================

CURRENT_DIRECTORY = Path(
    __file__
).resolve().parent


RUNTIME_GRAPH_MERMAID_FILE = (
    CURRENT_DIRECTORY
    / "task4_runtime_graph.mmd"
)


RUNTIME_GRAPH_PNG_FILE = (
    CURRENT_DIRECTORY
    / "task4_runtime_graph.png"
)


# ============================================================
# 2. VPN KNOWLEDGE BASE
#
# This matches the Task 4 example supplied in the assignment.
# ============================================================

VPN_KNOWLEDGE_BASE = [
    "Restart the VPN Client",
    "Clear the DNS Cache",
    "Reset the Network Adapter",
]


# ============================================================
# 3. GRAPH SETTINGS
# ============================================================

MAX_TROUBLESHOOTING_STEPS = len(
    VPN_KNOWLEDGE_BASE
)

GRAPH_RECURSION_LIMIT = 20


# ============================================================
# 4. DEFINE LANGGRAPH SHARED STATE
# ============================================================

class HelpdeskState(TypedDict, total=False):
    """
    Shared state for the IT Helpdesk LangGraph.

    Input:
        issue

    Agent 1:
        diagnosis
        agent1_message
        enough_information
        diagnosis_count

    Agent 2:
        troubleshooting_steps
        next_step_index
        agent2_message

    Final:
        final_response

    Audit:
        execution_trace
    """

    issue: str

    diagnosis: str

    agent1_message: str

    agent2_message: str

    troubleshooting_steps: list[str]

    next_step_index: int

    enough_information: bool

    diagnosis_count: int

    final_response: str

    execution_trace: list[str]


# ============================================================
# 5. VALIDATE ASSIGNMENT ISSUE
# ============================================================

def validate_issue(
    issue: str
) -> None:
    """
    Task 4 supplied a VPN troubleshooting scenario.

    Avoid returning VPN guidance for an unrelated issue.
    """

    normalized_issue = (
        issue
        .lower()
        .strip()
    )


    if "vpn" not in normalized_issue:

        raise ValueError(
            "\nThis Task 4 implementation follows the "
            "VPN troubleshooting scenario in the "
            "assignment.\n\n"
            "Please enter a VPN-related issue.\n\n"
            "Example:\n"
            "I cannot connect to VPN after changing "
            "my password."
        )


# ============================================================
# 6. AGENT 1 - DIAGNOSIS AGENT
# ============================================================

def diagnosis_agent(
    state: HelpdeskState
) -> dict:
    """
    Agent 1 responsibilities:

    1. Analyze issue.
    2. Review troubleshooting collected so far.
    3. Determine whether another KB search is required.
    4. Stop the loop when sufficient information exists.
    """

    issue = state["issue"]


    troubleshooting_steps = list(
        state.get(
            "troubleshooting_steps",
            []
        )
    )


    diagnosis_count = (
        state.get(
            "diagnosis_count",
            0
        )
        + 1
    )


    execution_trace = list(
        state.get(
            "execution_trace",
            []
        )
    )


    print(
        "\n========================================"
    )

    print(
        "AGENT 1 EXECUTED"
    )

    print(
        "DIAGNOSIS AGENT"
    )

    print(
        "========================================"
    )


    print(
        "Diagnosis pass:",
        diagnosis_count
    )


    print(
        "Issue:",
        issue
    )


    print(
        "Troubleshooting steps collected:",
        len(troubleshooting_steps)
    )


    # --------------------------------------------------------
    # FIRST DIAGNOSIS
    # --------------------------------------------------------

    if len(troubleshooting_steps) == 0:

        diagnosis = (
            "VPN authentication issue detected."
        )


        agent1_message = (
            "VPN authentication issue detected. "
            "Need troubleshooting guidance."
        )


        enough_information = False


        trace_message = (
            f"Agent 1 pass {diagnosis_count}: "
            "VPN authentication issue detected; "
            "requested troubleshooting guidance"
        )


    # --------------------------------------------------------
    # MORE TROUBLESHOOTING REQUIRED
    # --------------------------------------------------------

    elif (
        len(troubleshooting_steps)
        <
        MAX_TROUBLESHOOTING_STEPS
    ):

        diagnosis = (
            "VPN troubleshooting is still in progress."
        )


        agent1_message = (
            "Need additional troubleshooting."
        )


        enough_information = False


        trace_message = (
            f"Agent 1 pass {diagnosis_count}: "
            "requested additional troubleshooting"
        )


    # --------------------------------------------------------
    # ENOUGH INFORMATION
    # --------------------------------------------------------

    else:

        diagnosis = (
            "Enough troubleshooting information "
            "has been collected."
        )


        agent1_message = (
            "Enough information collected. "
            "Generate final response."
        )


        enough_information = True


        trace_message = (
            f"Agent 1 pass {diagnosis_count}: "
            "enough information collected"
        )


    execution_trace.append(
        trace_message
    )


    print(
        "Diagnosis:",
        diagnosis
    )


    print(
        "Agent 1 response:",
        agent1_message
    )


    print(
        "Enough information:",
        enough_information
    )


    return {

        "diagnosis":
            diagnosis,

        "agent1_message":
            agent1_message,

        "enough_information":
            enough_information,

        "diagnosis_count":
            diagnosis_count,

        "execution_trace":
            execution_trace,
    }


# ============================================================
# 7. CONDITIONAL ROUTER
# ============================================================

def route_after_diagnosis(
    state: HelpdeskState
) -> Literal[
    "need_troubleshooting",
    "enough_information"
]:
    """
    Conditional decision after Agent 1.

    NO  -> Agent 2
    YES -> Final response
    """

    if state["enough_information"]:

        print(
            "\nRouting Decision:"
        )

        print(
            "Enough Information = YES"
        )

        print(
            "Agent 1 -> Generate Final Response"
        )

        return "enough_information"


    print(
        "\nRouting Decision:"
    )

    print(
        "Enough Information = NO"
    )

    print(
        "Agent 1 -> Agent 2"
    )


    return "need_troubleshooting"


# ============================================================
# 8. AGENT 2 - RESOLUTION AGENT
# ============================================================

def resolution_agent(
    state: HelpdeskState
) -> dict:
    """
    Agent 2 responsibilities:

    1. Search the simulated knowledge base.
    2. Return the next troubleshooting step.
    3. Return control to Agent 1.
    """

    troubleshooting_steps = list(
        state.get(
            "troubleshooting_steps",
            []
        )
    )


    next_step_index = state.get(
        "next_step_index",
        0
    )


    execution_trace = list(
        state.get(
            "execution_trace",
            []
        )
    )


    print(
        "\n========================================"
    )

    print(
        "AGENT 2 EXECUTED"
    )

    print(
        "RESOLUTION AGENT"
    )

    print(
        "========================================"
    )


    if (
        next_step_index
        >=
        len(VPN_KNOWLEDGE_BASE)
    ):

        raise RuntimeError(
            "There are no additional troubleshooting "
            "steps in the knowledge base."
        )


    # --------------------------------------------------------
    # SIMULATED KNOWLEDGE BASE SEARCH
    # --------------------------------------------------------

    troubleshooting_step = (
        VPN_KNOWLEDGE_BASE[
            next_step_index
        ]
    )


    troubleshooting_steps.append(
        troubleshooting_step
    )


    updated_step_index = (
        next_step_index
        + 1
    )


    agent2_message = (
        troubleshooting_step
    )


    execution_trace.append(
        f"Agent 2 search "
        f"{updated_step_index}: "
        f"{troubleshooting_step}"
    )


    print(
        "Knowledge Base Search:",
        updated_step_index
    )


    print(
        "Troubleshooting guidance:",
        troubleshooting_step
    )


    print(
        "Returning control to Agent 1."
    )


    return {

        "agent2_message":
            agent2_message,

        "troubleshooting_steps":
            troubleshooting_steps,

        "next_step_index":
            updated_step_index,

        "execution_trace":
            execution_trace,
    }


# ============================================================
# 9. FINAL RESPONSE NODE
# ============================================================

def final_response_node(
    state: HelpdeskState
) -> dict:
    """
    Generate the final response after Agent 1 determines
    sufficient troubleshooting information has been found.
    """

    troubleshooting_steps = (
        state[
            "troubleshooting_steps"
        ]
    )


    execution_trace = list(
        state.get(
            "execution_trace",
            []
        )
    )


    numbered_steps = "\n".join(

        f"{step_number}. {step}."

        for step_number, step
        in enumerate(
            troubleshooting_steps,
            start=1
        )
    )


    final_response = (
        "To resolve the VPN connectivity issue, "
        "please try the following:\n\n"
        f"{numbered_steps}\n\n"
        "If the issue persists after completing "
        "these steps, contact the IT Helpdesk "
        "for further investigation."
    )


    execution_trace.append(
        "Generated final troubleshooting response"
    )


    print(
        "\n========================================"
    )

    print(
        "FINAL RESPONSE NODE EXECUTED"
    )

    print(
        "========================================"
    )


    print(
        final_response
    )


    return {

        "final_response":
            final_response,

        "execution_trace":
            execution_trace,
    }


# ============================================================
# 10. CREATE STATEGRAPH
# ============================================================

graph_builder = StateGraph(
    HelpdeskState
)


# ============================================================
# 11. ADD AGENT NODES
# ============================================================

graph_builder.add_node(
    "diagnosis_agent",
    diagnosis_agent
)


graph_builder.add_node(
    "resolution_agent",
    resolution_agent
)


graph_builder.add_node(
    "final_response",
    final_response_node
)


# ============================================================
# 12. START -> AGENT 1
# ============================================================

graph_builder.add_edge(
    START,
    "diagnosis_agent"
)


# ============================================================
# 13. CONDITIONAL ROUTING
#
# Agent 1
#    |
#    +--- NO ---> Agent 2
#    |
#    +--- YES --> Final Response
# ============================================================

graph_builder.add_conditional_edges(

    "diagnosis_agent",

    route_after_diagnosis,

    {

        "need_troubleshooting":
            "resolution_agent",

        "enough_information":
            "final_response",
    }
)


# ============================================================
# 14. CREATE LOOP
#
# Agent 2 -> Agent 1
# ============================================================

graph_builder.add_edge(
    "resolution_agent",
    "diagnosis_agent"
)


# ============================================================
# 15. FINAL RESPONSE -> END
# ============================================================

graph_builder.add_edge(
    "final_response",
    END
)


# ============================================================
# 16. COMPILE LANGGRAPH
# ============================================================

helpdesk_graph = (
    graph_builder.compile()
)


# ============================================================
# 17. CLEAN TEXT FOR MERMAID
# ============================================================

def clean_mermaid_text(
    value: object
) -> str:
    """
    Simplify text before inserting it into Mermaid labels.
    """

    return (
        str(value)
        .replace("&", "and")
        .replace('"', "'")
        .replace("<", "")
        .replace(">", "")
        .replace("\n", " ")
        .replace(";", ",")
    )


# ============================================================
# 18. BUILD SIMPLE RUNTIME MERMAID
#
# This intentionally avoids complicated HTML and styling.
# It is simpler and more reliable for Mermaid PNG rendering.
# ============================================================

def build_runtime_mermaid(
    input_state: HelpdeskState,
    output_state: HelpdeskState
) -> str:
    """
    Create runtime Mermaid diagram after LangGraph execution.
    """

    issue = clean_mermaid_text(
        input_state["issue"]
    )


    diagnosis_count = (
        output_state[
            "diagnosis_count"
        ]
    )


    steps = (
        output_state[
            "troubleshooting_steps"
        ]
    )


    step1 = clean_mermaid_text(
        steps[0]
        if len(steps) > 0
        else "No Step"
    )


    step2 = clean_mermaid_text(
        steps[1]
        if len(steps) > 1
        else "No Step"
    )


    step3 = clean_mermaid_text(
        steps[2]
        if len(steps) > 2
        else "No Step"
    )


    runtime_mermaid = f"""
flowchart TD

    START([START])

    ISSUE["VPN Issue Reported: {issue}"]

    AGENT1["Agent 1: Diagnosis Agent - Passes: {diagnosis_count}"]

    DECISION{{"Enough Information?"}}

    AGENT2["Agent 2: Resolution Agent"]

    KB1["KB Step 1: {step1}"]

    KB2["KB Step 2: {step2}"]

    KB3["KB Step 3: {step3}"]

    FINAL["Generate Final Resolution"]

    END([END])

    START --> ISSUE

    ISSUE --> AGENT1

    AGENT1 --> DECISION

    DECISION -->|No| AGENT2

    AGENT2 --> KB1

    KB1 --> KB2

    KB2 --> KB3

    KB3 -->|Re-evaluate| AGENT1

    DECISION -->|Yes| FINAL

    FINAL --> END
"""


    return runtime_mermaid.strip()


# ============================================================
# 19. GENERATE RUNTIME GRAPH
#
# Same design principle as Tasks 1, 2, and 3:
#
# FIRST execute the workflow.
# THEN generate the graph.
# ============================================================

def generate_runtime_graph(
    input_state: HelpdeskState,
    output_state: HelpdeskState
) -> None:
    """
    Generate Mermaid and PNG after LangGraph execution.
    """

    print(
        "\n========================================"
    )

    print(
        "GENERATING TASK 4 RUNTIME GRAPH"
    )

    print(
        "========================================"
    )


    # --------------------------------------------------------
    # DISPLAY ACTUAL EXECUTION ROUTE
    # --------------------------------------------------------

    print(
        "\nExecution Path:"
    )


    print(
        "START"
    )


    print(
        "  -> VPN Issue Reported"
    )


    print(
        "  -> Agent 1"
    )


    for step_number, step in enumerate(
        output_state[
            "troubleshooting_steps"
        ],
        start=1
    ):

        print(
            f"  -> Loop {step_number}: "
            "Agent 1 -> Agent 2 "
            "-> Knowledge Base -> Agent 1"
        )


        print(
            "     KB Result:",
            step
        )


    print(
        "  -> Enough Information"
    )


    print(
        "  -> Generate Resolution"
    )


    print(
        "  -> END"
    )


    # --------------------------------------------------------
    # BUILD MERMAID
    # --------------------------------------------------------

    runtime_mermaid = (
        build_runtime_mermaid(
            input_state=input_state,
            output_state=output_state
        )
    )


    # --------------------------------------------------------
    # SAVE MERMAID
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
    # REMOVE OLD PNG
    # --------------------------------------------------------

    if RUNTIME_GRAPH_PNG_FILE.exists():

        try:

            RUNTIME_GRAPH_PNG_FILE.unlink()

        except OSError:

            pass


    # --------------------------------------------------------
    # GENERATE PNG
    #
    # No Pyppeteer.
    # No Chromium installation.
    # --------------------------------------------------------

    try:

        print(
            "\nGenerating runtime PNG..."
        )


        png_data = draw_mermaid_png(

            mermaid_syntax=
                runtime_mermaid,

            background_color=
                "white",

            padding=
                20,

            max_retries=
                5,

            retry_delay=
                2.0,
        )


        # ----------------------------------------------------
        # SAVE PNG
        # ----------------------------------------------------

        RUNTIME_GRAPH_PNG_FILE.write_bytes(
            png_data
        )


        print(
            "\nRuntime PNG generated successfully:"
        )


        print(
            RUNTIME_GRAPH_PNG_FILE
        )


        # ----------------------------------------------------
        # OPEN PNG ON WINDOWS
        # ----------------------------------------------------

        if os.name == "nt":

            try:

                os.startfile(
                    str(
                        RUNTIME_GRAPH_PNG_FILE
                    )
                )


                print(
                    "\nRuntime graph opened automatically."
                )


            except OSError as open_error:

                print(
                    "\nPNG was generated successfully, "
                    "but Windows could not open it."
                )


                print(
                    "Open error:",
                    open_error
                )


    except Exception as graph_error:

        print(
            "\nPNG rendering failed."
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
            "\nThe Mermaid graph is still available:"
        )


        print(
            RUNTIME_GRAPH_MERMAID_FILE
        )


# ============================================================
# 20. EXECUTE HELPDESK WORKFLOW
# ============================================================

def execute_helpdesk_workflow(
    issue: str
) -> HelpdeskState:
    """
    Execute the complete troubleshooting loop.
    """

    # --------------------------------------------------------
    # VALIDATE TASK INPUT
    # --------------------------------------------------------

    validate_issue(
        issue
    )


    # --------------------------------------------------------
    # CREATE INITIAL STATE
    # --------------------------------------------------------

    input_state: HelpdeskState = {

        "issue":
            issue,

        "diagnosis":
            "",

        "agent1_message":
            "",

        "agent2_message":
            "",

        "troubleshooting_steps":
            [],

        "next_step_index":
            0,

        "enough_information":
            False,

        "diagnosis_count":
            0,

        "final_response":
            "",

        "execution_trace":
            [],
    }


    # --------------------------------------------------------
    # DISPLAY INPUT STATE
    # --------------------------------------------------------

    print(
        "\n======================================"
    )

    print(
        "LANGGRAPH INPUT STATE"
    )

    print(
        "========================================"
    )


    print(
        "{"
    )


    print(
        f'  "issue": "{issue}",'
    )


    print(
        '  "troubleshooting_steps": [],'
    )


    print(
        '  "enough_information": false,'
    )


    print(
        '  "diagnosis_count": 0'
    )


    print(
        "}"
    )


    # ========================================================
    # EXECUTE LANGGRAPH
    # ========================================================

    print(
        "\nExecuting IT Helpdesk "
        "LangGraph loop..."
    )


    try:

        final_state = (
            helpdesk_graph.invoke(

                input_state,

                config={
                    "recursion_limit":
                        GRAPH_RECURSION_LIMIT
                }
            )
        )


    except GraphRecursionError as error:

        raise RuntimeError(
            "The troubleshooting graph exceeded "
            "the recursion limit. Check the "
            "loop termination condition."
        ) from error


    # ========================================================
    # DISPLAY EXECUTION TRACE
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "EXECUTION TRACE"
    )

    print(
        "========================================"
    )


    for trace_number, trace_message in enumerate(
        final_state[
            "execution_trace"
        ],
        start=1
    ):

        print(
            f"{trace_number}. "
            f"{trace_message}"
        )


    # ========================================================
    # DISPLAY FINAL STATE
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "FINAL LANGGRAPH STATE"
    )

    print(
        "========================================"
    )


    print(
        "Issue:"
    )


    print(
        final_state[
            "issue"
        ]
    )


    print(
        "\nDiagnosis:"
    )


    print(
        final_state[
            "diagnosis"
        ]
    )


    print(
        "\nTroubleshooting Steps:"
    )


    for step_number, step in enumerate(
        final_state[
            "troubleshooting_steps"
        ],
        start=1
    ):

        print(
            f"{step_number}. {step}"
        )


    print(
        "\nDiagnosis passes:",
        final_state[
            "diagnosis_count"
        ]
    )


    print(
        "Enough information:",
        final_state[
            "enough_information"
        ]
    )


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "FINAL RESPONSE TO USER"
    )

    print(
        "========================================"
    )


    print(
        final_state[
            "final_response"
        ]
    )


    # ========================================================
    # GENERATE GRAPH ONLY AFTER EXECUTION
    # ========================================================

    generate_runtime_graph(
        input_state=input_state,
        output_state=final_state
    )


    return final_state


# ============================================================
# 21. MAIN APPLICATION
# ============================================================

def main() -> None:
    """
    Run Week 4 Task 4.
    """

    print(
        "\n========================================"
    )


    print(
        "ASCEND PYTHON LANGGRAPH"
    )


    print(
        "WEEK 4 - TASK 4"
    )


    print(
        "IT HELPDESK TROUBLESHOOTING AGENT LOOP"
    )


    print(
        "========================================"
    )


    print(
        "\nAssignment Test Input:"
    )


    print(
        "I cannot connect to VPN after changing "
        "my password."
    )


    issue = input(
        "\nReport the IT issue: "
    ).strip()


    if not issue:

        raise ValueError(
            "The IT issue cannot be empty."
        )


    execute_helpdesk_workflow(
        issue=issue
    )


# ============================================================
# 22. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()


    except KeyboardInterrupt:

        print(
            "\n\nTask 4 stopped by user."
        )


    except Exception as application_error:

        print(
            "\n========================================"
        )


        print(
            "TASK 4 FAILED"
        )


        print(
            "========================================"
        )


        print(
            "Error type:",
            type(
                application_error
            ).__name__
        )


        print(
            "Error details:",
            application_error
        )
        print(
            "========================================"
        )