# Embeddings: Turning Meaning into Numbers

## What is an embedding?
An embedding is a list of numbers (a vector) that represents the meaning of a piece of text. A small embedding model might turn any sentence into a list of 384 numbers. Texts with similar meanings get vectors that point in similar directions, even if they use completely different words. "How do I fix a flat tyre?" and "repairing a punctured bike wheel" end up close together.

## The map analogy
Imagine a giant map where every sentence is a dot. Sentences about cooking cluster in one region, sentences about football in another, and sentences about space travel somewhere else. An embedding is the coordinates of a sentence on that map. Real embeddings have hundreds of dimensions instead of two, which is impossible to picture, so visualizers squash them down to 2D using techniques like Principal Component Analysis (PCA). Squashing loses information, so distances on a 2D picture are only approximate.

## Measuring similarity
To compare two embeddings we usually use cosine similarity. It measures the angle between two vectors: a score near 1 means they point the same way (very similar meaning), a score near 0 means they are unrelated. Cosine similarity ignores the length of the vectors and only cares about direction.

## How embedding models learn
Embedding models are neural networks trained on huge numbers of text pairs. During training they are rewarded for placing related pairs (a question and its answer, a title and its article) close together and unrelated pairs far apart. This is called contrastive learning.

## Where embeddings are used
Embeddings power semantic search, recommendation systems, clustering of documents, duplicate detection, and the retrieval step of RAG systems. They are much cheaper to compute than running a full large language model.
