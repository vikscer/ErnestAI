from openai import OpenAI
from src.utils.config_loader import load_config
from datetime import datetime
import requests

# Load configuration
config = load_config()

# Set OpenAI API key
client = OpenAI(
    api_key=config['api_keys']['openai']
)

# Define the context for Ernest
ernest_context = config['openai_prompt']

# Initialize a list to store the conversation memory (last two exchanges)
conversation_memory = []


def get_time():
    """Returns the current time as a string."""
    return datetime.now().strftime("%H:%M")


def get_weather():
    """Fetches the current weather information. Replace this with a real API call."""
    return "sunny with a chance of rain"  # Placeholder text


def get_location():
    """Returns a static location, or implement a function to get dynamic location data."""
    return "New York City"


def update_conversation_memory(role, content):
    """Updates the conversation memory with the latest message."""
    # Append the new message to the conversation memory
    conversation_memory.append({"role": role, "content": content})

    # Keep only the last two exchanges (four messages total: user-assistant-user-assistant)
    if len(conversation_memory) > 4:
        conversation_memory.pop(0)  # Remove the oldest message


def generate_response(prompt):
    # Update the dynamic context for Ernest
    dynamic_context = ernest_context.replace("[time]", get_time())

    # Add the user prompt to conversation memory
    update_conversation_memory("user", prompt)

    # Construct the prompt with the dynamic context and the conversation memory
    messages = [{"role": "system", "content": dynamic_context}] + conversation_memory + [
        {"role": "user", "content": prompt}]

    print(messages)

    # Call the OpenAI GPT-4o Turbo chat model
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
    )

    # Extract the response text
    response_text = response.choices[0].message.content

    # Add the assistant's response to the conversation memory
    update_conversation_memory("assistant", response_text)

    # Return the final response
    return response_text
