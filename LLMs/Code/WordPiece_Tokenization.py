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
# print(splits)

# print('\n')

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
    # print(f'{key}:{pair_scores[key]}')
    if i>=5:
        break


#@ Finding the pair with the best score:
best_pair=''
max_score=None
for pair, score in pair_scores.items():
    if max_score is None or max_score<score:
     best_pair=pair
     max_score=score

# print(best_pair, max_score)


#@ Merging the pairs:
def merge_pair(a, b, splits):
    for word in words_freq:
        split=splits[word]
        if len(split)==1:
            continue

        i=0
        while i<len(split)-1:
            if split[i]==a and split[i+1]==b:
                merge=a + b[2:] if b.startswith("##") else a+b
                split=split[:i] + [merge] + split[i+2:]
            
            else:
                i+=1
        splits[word]=split

    return splits


splits=merge_pair('C', '##h', splits)
# print(splits['Chelsea'])



#@ Creating a vocab 

vocab_size=70
while len(vocab)< vocab_size:
    scores=compute_pair_score(splits)
    best_pair, max_score='', None
    for pair, score in scores.items():
        if max_score is None or max_score < score:
            best_pair=pair
            max_score=score
    splits=merge_pair(*best_pair, splits)

    new_token=(
                best_pair[0] + best_pair[1][2:]
                if best_pair[1].startswith('##')
                else best_pair[0] + best_pair[1]
    )
    vocab.append(new_token)


# print(vocab)

#@ encoder:

def encode_word(word):
    tokens=[]
    while len(word)>0:
        i=len(word)
        while i>0 and word[:i] not in vocab:
            i-=1
        if i==0:
            return ['[UNK]']
        
        tokens.append(word[:i])
        word=word[i:]
        if len(word)>0:
            word=f'##{word}'
    
    return tokens


#@ function that tokenizes the text:
def tokenize(text):
    pre_tokenize_result=tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text=[word for word, offset in pre_tokenize_result]
    encode_words=[encode_word(word) for word in pre_tokenized_text]
    return sum(encode_words, [])


val=tokenize('My name is prijal Khadka')
print(val)