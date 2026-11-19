
from openai import OpenAI
from app.config import settings


def check_models():

    if not settings.GROQ_API_KEY:
        print("Groq API key is missing.")
        return

    client = OpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url=settings.XAI_BASE_URL
    )

    models = client.models.list()

    print("\nAvailable Groq models:\n")

    for model in models.data:
        print(model.id)


if __name__ == "__main__":
    check_models()
