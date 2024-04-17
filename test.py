import math

def test(rank, layer, size):
  chunk = size >> layer + 1
  print(int(rank / chunk) * chunk)
  # print(rank >> layer + 1)
  return ((rank >> chunk) ^ 1) << layer

print(test(0, 0, 8))
  

# 