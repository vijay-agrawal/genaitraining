import openai

client = openai.OpenAI(
   api_key=""
)

# Using chat.completions instead of completions
chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",  # Note: curie is not available for chat completions
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

# Print the response text
print(chat_completion.choices[0].message.content)

# Print usage information
print(dict(chat_completion).get('usage'))

# Print the full response as JSON
print(chat_completion.model_dump_json(indent=2))


