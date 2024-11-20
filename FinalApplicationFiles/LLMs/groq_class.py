from groq import Groq
from dotenv import load_dotenv


load_dotenv()


class GrokModel:
    def __init__(self, api_key, model_name="llama3-8b-8192"):
        self.api_key = api_key
        self.model_name = model_name

    def generate_content(self, prompts):
        client = Groq(api_key=self.api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{prompts}",
                }
            ],
            model=self.model_name
        )

        message_content = chat_completion.choices[0].message.content
        return message_content