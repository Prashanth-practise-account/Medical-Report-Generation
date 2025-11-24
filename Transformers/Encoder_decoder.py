from transformers import pipeline
summarizer = pipeline("summarization", model="t5-small")

text = """
Transformers are modern neural networks that read entire sequences at once using self-attention.
They are faster and more accurate than RNNs and LSTMs for NLP tasks.
"""
result = summarizer(text)
print(result[0]['summary_text'])

# Translation from English to French
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

result = translator("Machine learning is the future of AI.")
print(result[0]['translation_text'])

# Question Answering using T5 model
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

context = """
The Transformer model was introduced in 2017 and revolutionized NLP by using attention mechanisms 
instead of recurrence. It became the foundation for models like BERT and GPT.
"""

question = "When was the Transformer model introduced?"

answer = qa(question=question, context=context)
print(answer['answer'])

