
from collections import Counter

def tag_count_feats(tree):

  c = Counter()

  for el in tree.iter():
    c[el.tag] += 1

  return c


def successful_feats(tree):

  c = Counter()

  for el in tree.iter():
    if el.get('successful'):
      c['num_successful'] += 1
    elif el.get('successful') == 0:
      c['num_unsuccessful'] += 1

  return c

def lines_count_feat(tree):

  c = Counter()

  for el in tree.iter():
    c['lines'] += 1

  return c
