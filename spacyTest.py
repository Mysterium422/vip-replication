import spacy

nlp = spacy.load("en_core_web_sm")

# Example sentences
sentence1 = "How long until my yellow card ends?"
sentence2 = "Will my yellow card end soon?"

# Process the sentences using spaCy
doc1 = nlp(sentence1)
doc2 = nlp(sentence2)

similarity = doc1.similarity(doc2)

print(similarity)