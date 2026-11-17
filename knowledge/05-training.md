# How Models Are Trained

## Pretraining: predicting the next token
A large language model starts with random weights. During pretraining it reads a huge amount of text and repeatedly tries to predict the next token. Each time it guesses wrong, a loss function measures how wrong it was, and the weights are nudged slightly to make the right answer more likely next time. After trillions of these tiny updates, the model becomes very good at predicting text, and along the way it picks up grammar, facts, and patterns of reasoning.

## Gradient descent and backpropagation
The method used to nudge the weights is gradient descent. Backpropagation calculates, for every weight, which direction to move it to reduce the loss. The learning rate controls how big each step is: too large and training becomes unstable, too small and it takes forever.

## Fine-tuning
After pretraining, a model is good at continuing text but not necessarily at following instructions. Fine-tuning trains it further on examples of good conversations, such as a question followed by a helpful answer.

## Learning from feedback
Models are then often improved using feedback on their answers. In reinforcement learning from human feedback (RLHF), people compare pairs of answers, a reward model learns their preferences, and the language model is trained to produce answers that score highly. Some approaches use a written set of principles, sometimes called a constitution, to guide this feedback.

## Training data cutoff
A model only knows about the world up to the point its training data was collected. This is called the knowledge cutoff. To answer questions about newer or private information, a model needs that information supplied in its prompt, which is exactly what RAG does.
