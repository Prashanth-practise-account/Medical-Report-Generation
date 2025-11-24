from transformers import pipeline
# test generation
# generator=pipeline("text-generation",model="gpt2")
# result = generator("Once upon a time in Bangalore,")
# print(result[0]['generated_text'])

# code generation
# code_gen = pipeline("text-generation", model="microsoft/CodeGPT-small-py")
#
# prompt = "# Python program to find factorial"
# result = code_gen(prompt, max_length=50, num_return_sequences=1)
# print(result[0]['generated_text'])

#  chat bot =  it give the replay
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")
user_input = "Hello, how are you?"
response = chatbot(user_input, max_length=50, num_return_sequences=1)
print(response[0]['generated_text'])