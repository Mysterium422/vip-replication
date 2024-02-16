import wikipedia
import re
import random

company = "Microsoft"
company_title = wikipedia.search(company + " company")[0]
print(company_title)
wiki = wikipedia.page(title=company_title, auto_suggest=False)

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

random.shuffle(positive_sentences)
random.shuffle(negative_sentences)

def startswithcompany(string):
  return string.lower().startswith(company.lower()) or string.lower().startswith(company_title.lower()) or string.lower().startswith("the company")

if len(negative_sentences) > len(positive_sentences):
  pos_company_count = sum(1 for string in positive_sentences if startswithcompany(string))
  neg_company_count = sum(1 for string in negative_sentences if startswithcompany(string))
  while len(negative_sentences) > len(positive_sentences):
    if neg_company_count > pos_company_count:
      for i, sentence in enumerate(negative_sentences):
        if startswithcompany(sentence):
          del negative_sentences[i]
          neg_company_count -= 1
          break
        if i == len(negative_sentences) - 1:
          amtToRemove = len(negative_sentences) - len(positive_sentences)
          negative_sentences[:-amtToRemove]
          break
    else:
      for i, sentence in enumerate(negative_sentences):
        if not startswithcompany(sentence):
          del negative_sentences[i]
          break
        if i == len(negative_sentences) - 1:
          amtToRemove = len(negative_sentences) - len(positive_sentences)
          negative_sentences[:-amtToRemove]
          break
elif len(positive_sentences) > len(negative_sentences):
  pos_company_count = sum(1 for string in positive_sentences if startswithcompany(string))
  neg_company_count = sum(1 for string in negative_sentences if startswithcompany(string))
  print(pos_company_count, neg_company_count)
  while len(positive_sentences) > len(negative_sentences):
    if pos_company_count > neg_company_count:
      for i, sentence in enumerate(positive_sentences):
        if startswithcompany(sentence):
          del positive_sentences[i]
          pos_company_count -= 1
          break
        if i == len(positive_sentences) - 1:
          amtToRemove = len(positive_sentences) - len(negative_sentences)
          positive_sentences[:-amtToRemove]
          break
    else:
      for i, sentence in enumerate(positive_sentences):
        if not startswithcompany(sentence):
          del positive_sentences[i]
          break
        if i == len(positive_sentences) - 1:
          amtToRemove = len(positive_sentences) - len(negative_sentences)
          positive_sentences[:-amtToRemove]
          break


print(len(positive_sentences))
print(sum(1 for string in positive_sentences if startswithcompany(string)))
print(len(negative_sentences))
print(sum(1 for string in negative_sentences if startswithcompany(string)))