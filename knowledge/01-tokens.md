# Tokens: How AI Reads Text

## What is a token?
A language model never sees letters or whole sentences directly. Before any processing, text is split into small pieces called tokens. A token is often a word, part of a word, or a punctuation mark. For example, "unbelievable" might become three tokens: "un", "believ", and "able". A rough rule of thumb for English is that one token is about four characters, or about three quarters of a word.

## Why not just use words or letters?
Using whole words would need an enormous vocabulary and could not handle new words, typos, or names. Using single letters would make sequences very long and slow to process. Subword tokens are a compromise: common words get a single token, while rare words are built from smaller pieces. The most common method for building this vocabulary is called Byte Pair Encoding (BPE), which repeatedly merges the most frequent pairs of characters into new tokens.

## Token IDs
Each token in the vocabulary has a number called its token ID. The model actually works with these numbers, not with text. The sentence "I like cats" might become a list such as [40, 588, 19987]. When the model writes an answer, it produces token IDs one at a time, which are then turned back into text.

## Context window
The context window is the maximum number of tokens a model can look at in one go. It includes the instructions, any documents, the conversation so far, and the answer being written. If something does not fit in the context window, the model cannot see it at all. This limit is one of the main reasons Retrieval-Augmented Generation (RAG) exists: instead of pasting a whole library into the prompt, we retrieve only the most relevant pieces.

## Why tokens matter to you
AI services usually charge per token, and speed is measured in tokens per second. Languages other than English, code, and unusual words often use more tokens for the same amount of meaning.
