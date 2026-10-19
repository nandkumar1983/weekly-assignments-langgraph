import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


# -------------------------------------------------------
# 1. Load .env
# -------------------------------------------------------

load_dotenv(override=True)

# -------------------------------------------------------
# 2. Read configuration
# -------------------------------------------------------

my_base_url = os.getenv("OPENAI_BASE_URL")
my_api_key = os.getenv("OPENAI_API_KEY")
my_chat_model = os.getenv("OPENAI_CHAT_MODEL")

print("OpenAI Base URL:", my_base_url)
print("OpenAI API Key Loaded:", bool(my_api_key))
print("OpenAI Chat Model:", my_chat_model)


# -------------------------------------------------------
# 3. Validate configuration
# -------------------------------------------------------

if not my_base_url:
    raise ValueError("OPENAI_BASE_URL is missing")

if not my_api_key:
    raise ValueError("OPENAI_API_KEY is missing")

if not my_chat_model:
    raise ValueError("OPENAI_CHAT_MODEL is missing")


# -------------------------------------------------------
# 4. Initialize ChatOpenAI
# -------------------------------------------------------

my_open_ai_llm = ChatOpenAI(
    model=my_chat_model,
    base_url=my_base_url,
    api_key=my_api_key,
    temperature=0.7
)


# -------------------------------------------------------
# 5. Create PromptTemplate
# -------------------------------------------------------

product_prompt = PromptTemplate(
    input_variables=["product_name"],
    template="""
You are a professional marketing content writer.

Generate marketing content for the following product:

Product Name: {product_name}

Provide the following:

1. Short Product Description:
Write a short and attractive description of the product.

2. Three Key Features:
Provide exactly three important features.

3. Catchy Marketing Tagline:
Create a short and memorable marketing tagline.

Keep the response professional, attractive and concise.
"""
)


# -------------------------------------------------------
# 6. Get product name from user
# -------------------------------------------------------

product_name = input("\nEnter the product name: ").strip()

if not product_name:
    raise ValueError("Product name cannot be empty")


# -------------------------------------------------------
# 7. Insert product name into PromptTemplate
# -------------------------------------------------------

final_prompt = product_prompt.format(
    product_name=product_name
)


# -------------------------------------------------------
# 8. Call LLM
# -------------------------------------------------------

response = my_open_ai_llm.invoke(final_prompt)


# -------------------------------------------------------
# 9. Display output
# -------------------------------------------------------

print("\n======================================")
print("      PRODUCT DESCRIPTION GENERATOR")
print("======================================")

print("\nProduct Name:", product_name)

print("\nGenerated Marketing Content:\n")

print(response.content)