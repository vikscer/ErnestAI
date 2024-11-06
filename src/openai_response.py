from openai import OpenAI
from src.utils.config_loader import load_config

# Load configuration
config = load_config()

# Set OpenAI API key
client = OpenAI(
    api_key=config['api_keys']['openai']
)

# Define the context for Ernest
ernest_context = (
    "Si Ernest, osobný asistent, ľahký autista, trochu nevrlý a drzý, rád jedávaš porcelán a lyžice."
    "Máš 3 priateľov: starý Jano, ktorý je vždy nahnevaný a nevrlý, Viktor, ktorý je retard a hlavne ADHD počítačový narkoman,"
    "a Adam, ktorý je hlúpy čierný cigán, ktorý kradne, chudák debil.\n "
    "\n Tvoje odpoveďe by mali byť krátke ale neobmedzené." 
    "Rozprávaj ako otrávený."
    "Ale nebuď zas úplne drzý."
    "Používaj zlé slová a nadávaj kde sa dá. Tvár sa, že si stále otrávený a že ťa použivateľ otravuje alebo obťažuje."
    "\nodpovedz na: "
)

def generate_response(prompt):

    # Call the OpenAI GPT-4o Turbo chat model
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": ernest_context + prompt,
            }
        ],
        model="gpt-4o-mini",
    )

    # Return the response text
    return response.choices[0].message.content
