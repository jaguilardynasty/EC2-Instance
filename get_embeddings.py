# import openai
# import asyncio
# from openai.embeddings_utils import get_embedding

# # Set your OpenAI API key here
# openai.api_key = "REDACTED_OPENAI_KEY"

# async def get_embeddings(sentences, engine='text-embedding-ada-002'):
#     loop = asyncio.get_running_loop()
#     embeddings = await asyncio.gather(*[loop.run_in_executor(None, get_embedding, sentence, engine) for sentence in sentences])
#     return embeddings

# def split_into_sentences(text):
#     return [line.strip() for line in text.splitlines() if line.strip()]

# async def get_text_embeddings(input_text, engine='text-embedding-ada-002'):
#     sentences = split_into_sentences(input_text)
#     return await get_embeddings(sentences, engine)

# # Modified function to accept a list of sentences
# async def get_text_embeddings_from_list(sentences, engine='text-embedding-ada-002'):
#     return await get_embeddings(sentences, engine)
