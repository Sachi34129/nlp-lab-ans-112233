# nlp-lab-ans-112233

1

import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer, TweetTokenizer, MWETokenizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

# Download necessary resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample text
text = "I'm learning NLP! NLTK's tools like tokenizers, stemmers, and lemmatizers are useful."

# 1. Tokenization
# Whitespace-based
whitespace_tokens = text.split()

# Punctuation-based using word_tokenize
punct_tokens = word_tokenize(text)

# Treebank tokenizer
treebank = TreebankWordTokenizer()
treebank_tokens = treebank.tokenize(text)

# Tweet tokenizer
tweet_tokenizer = TweetTokenizer()
tweet_tokens = tweet_tokenizer.tokenize(text)

# Multi-Word Expression Tokenizer (custom MWE)
mwe_tokenizer = MWETokenizer([('natural', 'language'), ('machine', 'learning')])
mwe_text = "I love natural language processing and machine learning."
mwe_tokens = mwe_tokenizer.tokenize(mwe_text.split())

# 2. Stemming
porter = PorterStemmer()
snowball = SnowballStemmer("english")
porter_stems = [porter.stem(word) for word in punct_tokens]
snowball_stems = [snowball.stem(word) for word in punct_tokens]

# 3. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word) for word in punct_tokens]

# Print all outputs
print("Whitespace Tokenization:", whitespace_tokens)
print("Punctuation Tokenization:", punct_tokens)
print("Treebank Tokenization:", treebank_tokens)
print("Tweet Tokenization:", tweet_tokens)
print("MWE Tokenization:", mwe_tokens)

print("\nPorter Stemmer:", porter_stems)
print("Snowball Stemmer:", snowball_stems)
print("Lemmatization:", lemmas)
