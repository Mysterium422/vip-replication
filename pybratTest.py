from pybrat.parser import BratParser, Entity, Event, Example, Relation

brat = BratParser(error="ignore")
# examples = brat.parse("SentiFM-Binary")
examples = brat.parse("example-data/corpora/BioNLP-ST_2011")
print(len(examples))


