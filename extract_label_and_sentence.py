# --*-- encoding:utf-8 --*--

import nltk
import numpy as np
import string

from nltk.tokenize import WordPunctTokenizer  

stopwords = nltk.corpus.stopwords.words('english')
punctuation = string.punctuation

# Read the file
def read_file(file):
  with open(file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  for i in range(len(lines)):
    lines[i] = lines[i].strip()
  return lines

def get_label_and_sentence(content):
  label, sentence = [], []
  for i in range(len(content)):
    seg = content[i].split()
    label.append(int(seg[0]))   #labels

    #take a line
    sent = ' '.join(seg[1:])
    words = WordPunctTokenizer().tokenize(sent)
    this_sentence = []
    for ele in words:
      if (ele not in punctuation) and (ele not in stopwords)
        this_sentence.append(ele)
    sentence.append(this_sentence)

  return label, sentence

def main():
  content = read_file("data/feature_raw.txt")
  label, sentence = get_label_and_sentence(content)   # Get labels and sentence list

  # Write label
  with open("data/label.txt", 'w', encoding='utf-8') as f:
    for i in range(len(label)):
      f.write(str(label[i]) + '\n')  
  # Write lines
  with open('data/sentence.txt', 'w', encoding='utf-8') as f:
    for i in range(len(sentence)):
      for j in range(len(sentence[i])):
        f.write(sentence[i][j])
        if j != len(sentence[i])-1:
          f.write('\t')
      f.write('\n')


if __name__ == '__main__':
  main()