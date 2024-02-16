import wikipedia
import re

wiki = wikipedia.page(title="Microsoft", auto_suggest=False)
# print(wikipedia.search("Apple"))

text = wiki.content
split_lines = (text.split("\n"))

event_section_keywords = ["history", "creation", "leadership", "corporate", "acquisitions", "growth", "finance", "financial", "lawsuits", "litigation", "legal"]
date_pattern = r'^(On|In|By|As of)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
positive_sentences = []
negative_sentences = []

event_section = False
power = 2

for line in split_lines:
  if line.startswith("== "):
    split_words = line.lower().split(" ")
    if len((set(event_section_keywords) & set(split_words))) > 0:
      event_section = True
    else:
      event_section = False
  elif '.' in line:
    if event_section:
      sentences = line.split(".")
      for sentence in sentences:
        sentence = sentence.lstrip()
        if (len(sentence) < 5):
          continue 
        match = re.match(date_pattern, sentence)
        if match:
          sentence = sentence[match.end():]
          while sentence.startswith(" ") or sentence.startswith(","):
            sentence = sentence[1:]
          positive_sentences.append(sentence)
    else:
        sentences = line.split(".")
        for sentence in sentences:
          sentence = sentence.lstrip()
          if (len(sentence) < 5):
            continue 
          match = re.match(date_pattern, sentence)
          if not match:
            negative_sentences.append(sentence)

print(len(positive_sentences))
print(len(negative_sentences))