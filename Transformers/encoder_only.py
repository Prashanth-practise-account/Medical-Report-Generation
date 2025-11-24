from transformers import pipeline

# sentiment analyze

classifier = pipeline("sentiment-analysis", model="bert-base-uncased")
result = classifier("I love learning about AI and Transformers!")
print(result)

# test classification
classifier = pipeline("text-classification", model="bert-base-uncased")
text = "The movie was amazing and full of suspense!"
result = classifier(text)

print(result)
# named entity recognization (NER)

ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
text = "Barack Obama was born in Hawaii and served as President of the United States."
result = ner(text)
for entity in result:
    print(entity)