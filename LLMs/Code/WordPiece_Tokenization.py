corpus=[
    "My Name is Prijal Khadka",
    "I am currently pursuing Engineering in IT",
    "I live in Godawari, Lalitpur",
    "Chelsea is my Favourite Football Club", 
    "I listen to Post Malone When it comes to English Songs"
]


#@ Step 1:
"""
- pre-tokenize corpus into words
- since, WPE is used by BERT, we will use 'bert-base-cased' tokenizer
"""

from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained("bert-base-cased")


#@ Computing frequency of each word in corpuses:
from collections import defaultdict

words_freq=defaultdict(int)
for text in corpus:
    words_with_offset=tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words=[word for word, offset in words_with_offset]
    for word in new_words:
        words_freq[word]+=1
# print(words_freq)

#@ Working for alphabet:
alphabet=[]
for word in words_freq.keys():
    if word[0] not in alphabet:
        alphabet.append(word[0])

    for letter in word[1:]:
        if f'##{letter}' not in alphabet:
            alphabet.append(f'##{letter}')

# print(alphabet)


#@ Adding special tokens used specially by bert model:
vocab=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + alphabet.copy()
# print(vocab)

splits={
    word: [c if i==0 else f'##{c}' for i, c in enumerate(word)]
    for word in words_freq.keys()
}
print(splits)

print('\n')

#@Computing pair score:
def compute_pair_score(splits):
    letter_freq=defaultdict(int)
    pair_freq=defaultdict(int)

    for word, freq in words_freq.items():
        split=splits[word]
        if len(split)==1:
            letter_freq[split[0]]+=freq
            continue

        for i in range(len(split)-1):
            pair=(split[i], split[i+1])
            letter_freq[split[i]]+=freq
            pair_freq[pair]+=freq
        letter_freq[split[-1]]+=freq

    scores={
        pair: freq / (letter_freq[pair[0]] * letter_freq[pair[1]])
        for pair, freq in pair_freq.items()
    }

    return scores


#@ Testing:
pair_scores=compute_pair_score(splits)
for i, key in enumerate(pair_scores.keys()):
    print(f'{key}:{pair_scores[key]}')
    if i>=5:
        break