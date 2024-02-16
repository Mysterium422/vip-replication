import wikipedia
import re
import random

def getWikiPage(company):
  company_title = wikipedia.search(company + " company")[0]
  return wikipedia.page(title=company_title, auto_suggest=False)

def readFileLines(filepath):
  with open(filepath, "r") as file:
    return [line.strip() for line in file.readlines()]

def writeFileLines(filepath, array, overwrite=False):
  with open(filepath, "w" if overwrite else "a") as file:
    for line in array:
      file.write(str(line) + "\n")

event_section_keywords = ["history", "creation", "leadership", "corporate", "acquisitions", "growth", "finance", "financial", "lawsuits", "litigation", "legal"]
def isEventSection(split_words):
  return len((set(event_section_keywords) & set(split_words))) > 0

def lstriptrash(string):
  while string.startswith(" ") or string.startswith(","):
    string = string[1:]
  return string

def getDataFromCompany(company):
  wiki = getWikiPage(company)
  text = wiki.content
  split_lines = (text.split("\n"))

  event_section_keywords = ["history", "creation", "leadership", "corporate", "acquisitions", "growth", "finance", "financial", "lawsuits", "litigation", "legal"]
  date_pattern = r'^(On|In|By|As of)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})'
  positive_sentences = []
  negative_sentences = []

  event_section = False

  for line in split_lines:
    if line.startswith("== "):
      split_words = line.lower().split(" ")
      if isEventSection(split_words):
        event_section = True
      else:
        event_section = False
    elif '.' in line:
      if event_section:
        sentences = line.split(".")
        for sentence in sentences:
          sentence = lstriptrash(sentence)
          if (len(sentence) < 25):
            continue 
          match = re.match(date_pattern, sentence)
          if match:
            sentence = sentence[match.end():]
            sentence = lstriptrash(sentence)
            positive_sentences.append((sentence, int(match.group(3))))
      else:
          sentences = line.split(".")
          for sentence in sentences:
            sentence = lstriptrash(sentence)
            if (len(sentence) < 25):
              continue 
            match = re.match(date_pattern, sentence)
            if not match:
              negative_sentences.append(sentence)

  random.shuffle(positive_sentences)
  random.shuffle(negative_sentences)

  def startswithcompany(string):
    return string.lower().startswith(company.lower()) or string.lower().startswith(wiki.title.lower()) or string.lower().startswith("the company")

  if len(negative_sentences) > len(positive_sentences):
    pos_company_count = sum(1 for string, year in positive_sentences if startswithcompany(string))
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
    pos_company_count = sum(1 for string, year in positive_sentences if startswithcompany(string))
    neg_company_count = sum(1 for string in negative_sentences if startswithcompany(string))
    while len(positive_sentences) > len(negative_sentences):
      if pos_company_count > neg_company_count:
        for i, (sentence, year) in enumerate(positive_sentences):
          if startswithcompany(sentence):
            del positive_sentences[i]
            pos_company_count -= 1
            break
          if i == len(positive_sentences) - 1:
            amtToRemove = len(positive_sentences) - len(negative_sentences)
            positive_sentences[:-amtToRemove]
            break
      else:
        for i, (sentence, year) in enumerate(positive_sentences):
          if not startswithcompany(sentence):
            del positive_sentences[i]
            break
          if i == len(positive_sentences) - 1:
            amtToRemove = len(positive_sentences) - len(negative_sentences)
            positive_sentences[:-amtToRemove]
            break

  pos_test = [sentence for sentence, year in positive_sentences if year >= 2023]
  pos_train = [sentence for sentence, year in positive_sentences if year < 2023]
  random.shuffle(negative_sentences)
  neg_test = negative_sentences[:len(pos_test)]
  neg_train = negative_sentences[len(pos_test):]
  return pos_test, pos_train, neg_test, neg_train

def main():
  companies = readFileLines("companies.txt")
  for idx, company in enumerate(companies[:50], start=0):
    print(idx)
    pos_test, pos_train, neg_test, neg_train = getDataFromCompany(company)
    writeFileLines("positive_test.txt", pos_test, False)
    writeFileLines("positive_train.txt", pos_train, False)
    writeFileLines("negative_test.txt", neg_test, False)
    writeFileLines("negative_train.txt", neg_train, False)
    
if __name__ == "__main__":
    main()