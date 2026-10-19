import os

from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from dba_persona import DBA_SYSTEM_PROMPT

load_dotenv()

print("MODEL =", repr(os.getenv("MODEL_NAME")))
print("BASE_URL =", repr(os.getenv("BASE_URL")))
_key = os.getenv("API_KEY")
print("KEY PREFIX =", _key[:10] if _key else None)

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)


def main():

    print("=" * 60)
    print("DATABASE ADMINISTRATOR ASSISTANT")
    print("=" * 60)

    while True:

        question = input(
            "\nEnter your database question (type exit to quit): "
        )

        if question.lower() in ["exit", "quit", "q"]:
            print("Closing DBA Assistant...")
            break

        try:

            system_msg = SystemMessage(
                content=DBA_SYSTEM_PROMPT
            )

            human_msg = HumanMessage(
                content=question
            )

            response = client.chat.completions.create(
                model=os.getenv("MODEL_NAME"),
                messages=[
                    {
                        "role": "system",
                        "content": system_msg.content
                    },
                    {
                        "role": "user",
                        "content": human_msg.content
                    }
                ]
            )

            print("\nDBA Response:\n")
            print(response.choices[0].message.content)

        except Exception as ex:
            print("\nError:")
            print(ex)


if __name__ == "__main__":
    main()