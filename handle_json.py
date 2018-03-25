# --*-- encoding:utf-8 --*--

import json

def main():
  dict = {}
  select = {}   # select the characters

  with open('lines.json', 'r', encoding='utf-8') as f:
    text = json.load(f)
  for i in range(len(text)):
    name = text[i]['character']
    if name not in dict:
      dict[name] = 1
    else:
      dict[name] += 1
  line_left = 0
  for k,v in dict.items():
    if v > 1000:    # pick only the main characters who has more than 1000 lines
      select[k] = v
      line_left += v
  print(select)

  #print(select)
  print("Now there are {}% lines left, {} people left.".format(line_left/28877*100, len(select)))

  # label with numbers
  label = {}
  now_label = 0
  for k,v in select.items():
    if k in label:
      continue
    label[k] = now_label
    now_label += 1

  # write it to feature_raw.txt file (label + sentence)
  with open('data/feature_raw.txt', 'w', encoding='utf-8') as f:
    for i in range(len(text)):
      name = text[i]['character']
      sentence = text[i]['text']
      if name in select:
        f.write(str(label[name]) + '\t')
        f.write(sentence + '\n')

if __name__ == '__main__':
  main()