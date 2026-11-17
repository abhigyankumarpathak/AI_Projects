# Hallucinations and Limitations

## What is a hallucination?
A hallucination is when an AI model states something false with confidence, such as inventing a quote, a statistic, or a book that does not exist. It happens because the model is trained to produce likely-sounding text, and a plausible-sounding falsehood can look just as likely as the truth.

## Why it happens
The model has no built-in fact-checker. If its training data was thin on a topic, or the question is about something after its knowledge cutoff, it may fill the gap with a guess. Ambiguous questions and pressure to always give an answer make this worse.

## Reducing hallucinations
Grounding the model in real sources with RAG, asking it to cite those sources, allowing it to say "I don't know", and checking important facts independently all help. Lower sampling temperature can also make answers more conservative.

## Bias
Models learn from human-written text, which contains human biases. Without care, a model can repeat stereotypes or present one viewpoint as the only one. Developers test for and reduce bias, but no model is perfectly neutral.

## Using AI well as a student
Treat AI as a study partner, not an oracle. Ask it to explain its reasoning, check claims against your textbook or trusted sources, and use it to understand ideas rather than to replace your own thinking.
