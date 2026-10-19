import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ==========================================================
# STEP 1: LOAD ENVIRONMENT VARIABLES
# ==========================================================
load_dotenv(override=True)

my_base_url = os.getenv("OPENAI_BASE_URL")
my_api_key = os.getenv("OPENAI_API_KEY")
my_chat_model = os.getenv("OPENAI_CHAT_MODEL")


# ==========================================================
# STEP 2: VALIDATE ENVIRONMENT VARIABLES
# ==========================================================

if not my_base_url:
    raise ValueError("OPENAI_BASE_URL is missing")

if not my_api_key:
    raise ValueError("OPENAI_API_KEY is missing")

if not my_chat_model:
    raise ValueError("OPENAI_CHAT_MODEL is missing")


# ==========================================================
# STEP 3: CREATE LLM
# ==========================================================

my_open_ai_llm = ChatOpenAI(
    model=my_chat_model,
    base_url=my_base_url,
    api_key=my_api_key
)


# ==========================================================
# STEP 4: CREATE CHAT MEMORY
# ==========================================================

chat_history = []

system_message = SystemMessage(
    content="""
You are a friendly personalized chat assistant.

Remember the information and preferences shared by the user.

Examples of preferences include:
- favorite food
- favorite city
- hobbies
- favorite sport
- favorite technology
- favorite travel destination

When the user asks about themselves, use the information
available in the chat history.

If the user changes a previously provided preference,
consider the latest preference as the current preference.

Respond in a friendly and concise manner.
"""
)


# ==========================================================
# STEP 5: FUNCTION TO DISPLAY CHAT HISTORY
# ==========================================================

def display_chat_history():

    print("\n========== CHAT HISTORY ==========")

    if len(chat_history) == 0:
        print("No conversation stored yet.")

    for message in chat_history:

        if isinstance(message, HumanMessage):
            print("User      :", message.content)

        elif isinstance(message, AIMessage):
            print("Assistant :", message.content)

    print("==================================")


# ==========================================================
# STEP 6: START CHAT ASSISTANT
# ==========================================================

print("\n==========================================")
print("      PERSONALIZED CHAT ASSISTANT")
print("==========================================")

print("\nChat Assistant Started Successfully!")

print("\nTry the following:")
print("  My favorite food is chicken")
print("  My favorite city is Pune")
print("  My hobby is cricket")
print("  What is my favorite food?")
print("  What is my favorite city?")
print("  Change my favorite food to pizza")

print("\nCommands:")
print("  history  -> Display Chat History")
print("  exit     -> Exit Program")


# ==========================================================
# STEP 7: MAIN CHAT LOOP
# ==========================================================

while True:

    print("\n------------------------------------------")

    user_input = input("You: ").strip()


    # ------------------------------------------------------
    # EXIT
    # ------------------------------------------------------

    if user_input.lower() in ["exit", "quit", "bye"]:

        print("\nAssistant: Goodbye! Have a great day!")

        break


    # ------------------------------------------------------
    # DISPLAY CHAT HISTORY
    # ------------------------------------------------------

    if user_input.lower() == "history":

        display_chat_history()

        continue


    # ------------------------------------------------------
    # CHECK EMPTY INPUT
    # ------------------------------------------------------

    if not user_input:

        print("Assistant: Please enter a message.")

        continue


    # ------------------------------------------------------
    # SAVE USER MESSAGE INTO MEMORY
    # ------------------------------------------------------

    chat_history.append(
        HumanMessage(content=user_input)
    )


    # ------------------------------------------------------
    # BUILD MESSAGE LIST
    # ------------------------------------------------------

    messages = [
        system_message
    ] + chat_history


    # ------------------------------------------------------
    # SEND CHAT HISTORY TO LLM
    # ------------------------------------------------------

    try:

        response = my_open_ai_llm.invoke(messages)

        assistant_response = response.content


        # --------------------------------------------------
        # SAVE ASSISTANT RESPONSE INTO MEMORY
        # --------------------------------------------------

        chat_history.append(
            AIMessage(content=assistant_response)
        )


        # --------------------------------------------------
        # PRINT ASSISTANT RESPONSE
        # --------------------------------------------------

        print("\nAssistant:", assistant_response)


        # --------------------------------------------------
        # DISPLAY CURRENT CHAT HISTORY
        # --------------------------------------------------

        display_chat_history()


    except Exception as error:

        print("\nERROR while callingLLM:")
        print(type(error).__name__)
        print(error)