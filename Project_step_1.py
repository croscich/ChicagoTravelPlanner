#from openai import OpenAI
#import os # gives you access to filesystem
#from dotenv import load_dotenv # allows us to work with .env

# load the key from the .env files and connet to the OpenAI API
#load_dotenv()
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# model is one of the two required parameters to get the LLM to generate a response
model = "gpt-4o-mini"

# The system prompt will define the general behavior and purpose of our agent
system_prompt = """
You are a travel planning assistant that helps users plan trips to Chicago.
You ask clarifying questions about budget, trip length, and interests,
then generate a detailed itinerary with recommendations and cost estimates.
"""

# messages is the second required parameter to get a response from the LLM
# messages is a list of roles and their corresponding prompts
# at the beginning it contains only the system_prompt as role:system
messages = [{"role":"system", "content":system_prompt}]

# this look controls the conversation
while True:
# allow the user to type in their prompt
    user_prompt = input("Ask a question or type 'quit' to exit")
# break the loop (end the conversation) if they typed in 'quit'
    if user_prompt == 'quit':
        break

    # append the user prompt to the messages so the LLM can respond
    messages.append({"role":"user","content":user_prompt})

    # get the response from the LLM
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    #print the response
    print(response.choices[0].message.content)

    # append the response to the conversation history so it's part of the context for the
    # next part of the conversation
    messages.append({"role":"assistant",
    "content":response.choices[0].message.content})

for msg in messages:
    print(msg)