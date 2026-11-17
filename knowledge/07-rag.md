# Retrieval-Augmented Generation (RAG)

## The idea
Retrieval-Augmented Generation combines a search system with a language model. Instead of relying only on what the model memorised during training, a RAG system first looks up relevant documents and then gives them to the model along with the question. The model writes its answer using that supplied material. It is like an open-book exam instead of a closed-book one.

## Step 1: Chunking
Documents are split into smaller passages called chunks, often a few hundred words each, sometimes with a small overlap so that sentences at the edges are not cut off from their context. Chunks that are too large dilute the meaning; chunks that are too small lose context.

## Step 2: Indexing
Each chunk is turned into an embedding and stored in an index, often a vector database. This happens once, ahead of time.

## Step 3: Retrieval
When a question arrives, it is embedded with the same model. The system compares the question's vector with every chunk's vector, usually with cosine similarity, and picks the top-k most similar chunks.

## Step 4: Augmentation
The retrieved chunks are pasted into the prompt, together with instructions such as "answer using only the sources below and cite them".

## Step 5: Generation
The language model reads the prompt and writes an answer grounded in the sources, ideally citing which chunk each claim came from.

## Why use RAG?
RAG lets a model answer questions about private, recent, or specialised information without retraining it. It reduces hallucinations because the model has the facts in front of it, and it makes answers checkable because they can point to sources. Its weak spot is retrieval itself: if the right chunk is not retrieved, the model cannot use it.
