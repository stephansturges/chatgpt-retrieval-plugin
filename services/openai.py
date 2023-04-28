from typing import List
import openai
import replicate

from tenacity import retry, wait_random_exponential, stop_after_attempt

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_embeddings(texts: List[str]) -> List[List[float]]:
    embeddings_list = []
    
    for text in texts:
        output = replicate.run(
            "replicate/all-mpnet-base-v2:a441e62645d851373aa8a000e1471e8ac2f7c61a4b4148984bc190fc0b4c1f03",
            input={"text": text }
        )
        
        embeddings = output[0]["embedding"]
        embeddings_list.append(embeddings)
        print("got results from replicate")
        print(str(embeddings))
    
    return [embeddings_list.tolist()]

# @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
# def get_embeddings(texts: List[str]) -> List[List[float]]:
#     """
#     Embed texts using OpenAI's ada model.

#     Args:
#         texts: The list of texts to embed.

#     Returns:
#         A list of embeddings, each of which is a list of floats.

#     Raises:
#         Exception: If the OpenAI API call fails.
#     """
#     # Call the OpenAI API to get the embeddings
#     response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")

#     # Extract the embedding data from the response
#     data = response["data"]  # type: ignore

#     # Return the embeddings as a list of lists of floats
#     return [result["embedding"] for result in data]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_chat_completion(
    messages,
    model="gpt-3.5-turbo",  # use "gpt-4" for better results
):
    """
    Generate a chat completion using OpenAI's chat completion API.

    Args:
        messages: The list of messages in the chat history.
        model: The name of the model to use for the completion. Default is gpt-3.5-turbo, which is a fast, cheap and versatile model. Use gpt-4 for higher quality but slower results.

    Returns:
        A string containing the chat completion.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    # call the OpenAI chat completion API with the given messages
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )

    choices = response["choices"]  # type: ignore
    completion = choices[0].message.content.strip()
    print(f"Completion: {completion}")
    return completion
