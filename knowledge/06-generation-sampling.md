# How a Model Writes an Answer

## One token at a time
A language model generates text autoregressively: it predicts one token, appends it to the input, and then predicts the next one, over and over. That is why answers appear word by word when they are streamed.

## Probabilities over the vocabulary
At each step the model outputs a score called a logit for every token in its vocabulary. A softmax function turns these scores into probabilities that add up to 1. The model might say "Paris" has a 92% chance of coming next, "the" 3%, and so on.

## Sampling and temperature
Choosing the next token from these probabilities is called sampling. Temperature controls how adventurous the choice is. A low temperature makes the model almost always pick the most likely token, which gives focused, repeatable answers. A high temperature flattens the probabilities, so less likely tokens get picked more often, which gives more varied and creative but less reliable text. Top-p (nucleus) sampling only considers the smallest set of tokens whose probabilities add up to p.

## Stopping
Generation stops when the model produces a special end-of-turn token, or when it reaches the maximum number of tokens allowed for the answer.

## Thinking before answering
Some modern models can produce internal reasoning before their final answer. Spending more tokens reasoning step by step often improves accuracy on hard problems such as maths and coding, at the cost of more time and tokens.
