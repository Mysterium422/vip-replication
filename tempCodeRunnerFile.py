# Example sentences
sentence1 = "I love coding "
sentence2 = "I love studying"

# Process the sentences using spaCy
doc1 = nlp(sentence1)
doc2 = nlp(sentence2)

similarity = doc1.similarity(doc2)

print(similarity)