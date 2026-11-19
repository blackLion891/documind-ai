
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
    APIStatusError
)

from app.config import settings
from app.utils.exceptions import (
    ConfigurationError,
    LLMServiceError
)
from app.utils.logger import get_logger
from app.prompts.system_prompts import GENERAL_ASSISTANT_PROMPT


logger = get_logger(__name__)


def generate_response(
    prompt: str,
    system_prompt: str = GENERAL_ASSISTANT_PROMPT,
    conversation_history: list | None = None
) -> str:

    if not settings.GROQ_API_KEY:
        raise ConfigurationError("Groq API key is missing.")

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    client = OpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url=settings.GROQ_BASE_URL,
        timeout=60.0,
        max_retries=2
    )

    try:

        logger.info("Sending request to Groq")

        if conversation_history is None:
            conversation_history = []

        user_message = {
            "role": "user",
            "content": prompt
        }

        request_history = conversation_history + [user_message]

        response = client.responses.create(
            model=settings.GROQ_MODEL,
            instructions=system_prompt,
            input=request_history
        )

        answer = response.output_text

        if not answer or not answer.strip():
            raise LLMServiceError("Groq returned an empty response.")

        conversation_history.append(user_message)
        conversation_history.extend(response.output)

        usage = response.usage

        if usage:
            logger.info("Input tokens: %s", usage.input_tokens)
            logger.info("Output tokens: %s", usage.output_tokens)
            logger.info("Total tokens: %s", usage.total_tokens)

        logger.info("Groq response received successfully")

        return answer.strip()

    except AuthenticationError as error:
        logger.exception("Groq authentication failed")
        raise LLMServiceError("Invalid Groq API credentials.") from error

    except RateLimitError as error:
        logger.exception("Groq rate limit reached")
        raise LLMServiceError("Groq API rate limit reached.") from error

    except APITimeoutError as error:
        logger.exception("Groq request timed out")
        raise LLMServiceError("Groq API request timed out.") from error

    except APIConnectionError as error:
        logger.exception("Unable to connect to Groq")
        raise LLMServiceError("Could not connect to Groq.") from error

    except APIStatusError as error:
        logger.exception("Groq API returned an error")
        raise LLMServiceError(
            f"Groq API returned status {error.status_code}."
        ) from error

    except LLMServiceError:
        raise

    except Exception as error:
        logger.exception("Unexpected LLM service error")
        raise LLMServiceError("Unexpected error generating response.") from error
