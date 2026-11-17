# Transformers and Attention

## The transformer architecture
Almost every modern large language model is a transformer, an architecture introduced in the 2017 paper "Attention Is All You Need". A transformer is a stack of identical blocks. Each block contains an attention layer, which lets tokens share information with each other, and a feed-forward layer, which processes each token on its own.

## What attention does
Attention lets each token decide which other tokens are most relevant to it. In the sentence "The animal didn't cross the street because it was too tired", attention helps the word "it" focus strongly on "animal" rather than "street". Each token creates three vectors: a query (what am I looking for?), a key (what do I contain?), and a value (what information do I pass on?). The query of one token is compared with the keys of all other tokens to produce attention weights, which are then used to mix together their values.

## Multi-head attention
Transformers run several attention operations in parallel, called heads. Different heads can specialise: one might track grammar, another might link pronouns to nouns, another might follow quotation marks.

## Positional information
Attention on its own does not know the order of tokens. Transformers add positional encodings so the model knows that "dog bites man" is different from "man bites dog".

## Why transformers won
Unlike older recurrent networks, which read text one word at a time, transformers process all tokens in parallel during training. This makes them much faster to train on modern GPUs, which allowed researchers to scale them to enormous sizes.
