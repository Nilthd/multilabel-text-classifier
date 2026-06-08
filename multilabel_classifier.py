"""
Design Requirement Multi-Label Classifier
Imagine you have thousands of engineering sentences like:
    "The system shall respond within 2ms under full load."

the job is to automatically tag each sentence with categories like:
    ✓ performance_test
    ✓ timing_requirement
    ✗ safety_test  (not relevant)

One sentence can have MULTIPLE tags at once — that's called multi-label classification.

HOW IT WORKS:
------------------------------
1. A pretrained encoder (like BERT) reads the sentence and turns it
   into a list of numbers that captures its meaning — like a fingerprint.

2. Our small neural network (the "head") looks at that fingerprint
   and outputs a score (0 to 1) for each of the 50 possible categories.

3. If a score is >= 0.5, we say "yes, this sentence belongs to that category."
   If a score is < 0.5, we say "no."

WHY SIGMOID AND NOT SOFTMAX?
------------------------------
- Softmax: forces all scores to add up to 1 — use when only ONE answer is correct
           (like image classification: is this a cat OR a dog?)
- Sigmoid: each score is independent — use when MULTIPLE answers can be correct
           (like here: a sentence can be BOTH a timing AND a performance requirement)

Author: Niloofar Tavahoodi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ── PART 1 & 2: The Classification Head ──────────────────────────────────────
#
# Think of the pretrained encoder (BERT) as a very smart reader.
# It reads a sentence and gives us back a rich description (768 numbers).
#
# But BERT doesn't know about OUR 50 categories — it was trained on general text.
# So we add a small "head" network on top that learns OUR specific categories.
#
# The head is a 2-layer MLP (Multi-Layer Perceptron):
#
#   Input:  768 numbers (BERT's summary of the sentence)
#     ↓
#   Layer 1: shrink from 768 → 384  (compress, find patterns)
#     ↓
#   ReLU: set any negative number to 0 (adds non-linearity — lets the model
#         learn complex patterns, not just straight lines)
#     ↓
#   Layer 2: shrink from 384 → 50   (one score per category)
#     ↓
#   Output: 50 raw scores called "logits" (can be any number, positive or negative)
#
# Later, sigmoid() turns those logits into probabilities between 0 and 1.

class MultiLabelClassificationHead(nn.Module):

    def __init__(self, hidden_dim: int, num_classes: int):
        """
        hidden_dim  : size of BERT's output vector (e.g. 768)
        num_classes : how many categories we're classifying into (e.g. 50)
        """
        super().__init__()

        # Layer 1: hidden_dim → hidden_dim // 2
        # Example: 768 → 384
        # Think of this as the model learning "what patterns matter"
        self.layer1 = nn.Linear(hidden_dim, hidden_dim // 2)

        # Layer 2: hidden_dim // 2 → num_classes
        # Example: 384 → 50
        # Think of this as the model saying "based on those patterns,
        # how likely is each of our 50 categories?"
        self.layer2 = nn.Linear(hidden_dim // 2, num_classes)

        # ReLU activation — applied between the two layers
        # It sets negative numbers to 0: f(x) = max(0, x)
        # Without this, two linear layers would just collapse into one.
        # ReLU lets the model learn curved/complex decision boundaries.
        self.relu = nn.ReLU()

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Takes BERT's CLS token (the sentence summary) and returns logits.

        Args:
            cls_token : (batch_size, hidden_dim)
                        — one vector per sentence in the batch

        Returns:
            logits    : (batch_size, num_classes)
                        — one raw score per category per sentence
                        — NOT probabilities yet (no sigmoid here)
                        — sigmoid is applied in the loss function during training,
                          and manually during inference
        """
        # Step 1: pass through first layer → compress to hidden_dim//2
        x = self.layer1(cls_token)   # (batch_size, hidden_dim//2)

        # Step 2: apply ReLU — kill negative values
        x = self.relu(x)             # (batch_size, hidden_dim//2)

        # Step 3: pass through second layer → one score per class
        x = self.layer2(x)           # (batch_size, num_classes)

        # Return raw logits — NOT sigmoid yet
        # BCEWithLogitsLoss (used in training) applies sigmoid internally,
        # which is numerically more stable than doing it manually here.
        return x


# ── PART 3: Training Loop ─────────────────────────────────────────────────────
#
# This function trains the model for ONE full pass over the training data.
# One full pass is called an "epoch".
#
# What happens in each step:
#
#   1. Take a batch of sentences (e.g. 32 at a time)
#   2. Feed them through BERT → get fingerprints
#   3. Feed fingerprints through our head → get 50 scores per sentence
#   4. Compare scores to the correct labels → compute loss
#      (loss = how wrong were we?)
#   5. Backpropagate → figure out which weights caused the error
#   6. Update weights → nudge them in the right direction
#   7. Repeat for next batch
#
# WHAT LOSS FUNCTION DO WE USE?
# BCEWithLogitsLoss = Binary Cross Entropy with Logits
#
# "Binary" because each class is a yes/no question independently.
# "With Logits" means it applies sigmoid internally (more numerically stable).
#
# Think of it as: for each of the 50 classes, the loss asks
# "how surprised are we by this prediction given the true answer?"
# Confident and correct = low loss. Confident and wrong = high loss.

def train_one_epoch(
    encoder:    nn.Module,
    head:       MultiLabelClassificationHead,
    dataloader: DataLoader,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
) -> float:
    """
    Trains encoder + head for one full pass over the dataset.

    Returns:
        average loss across all batches (lower = better)
    """
    # Put both models in training mode
    # (this enables things like dropout if present)
    encoder.train()
    head.train()

    # BCEWithLogitsLoss = perfect for multi-label problems
    # It applies sigmoid to our logits, then computes binary cross-entropy
    # for each class independently.
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0

    for input_ids, attention_mask, labels in dataloader:

        # Move data to GPU (if available) — same device as the model
        input_ids      = input_ids.to(device)       # (batch, seq_len)
        attention_mask = attention_mask.to(device)  # (batch, seq_len)
        labels         = labels.to(device).float()  # (batch, num_classes) — multi-hot

        # ── Step 1: zero out gradients from the previous batch ──
        # If we don't do this, gradients ACCUMULATE across batches,
        # which gives wrong updates. Always clear before each step.
        optimizer.zero_grad()

        # ── Step 2: forward pass through BERT ──
        # encoder returns (batch_size, seq_len, hidden_dim)
        # We only care about the CLS token — position 0 — which BERT
        # trains to be a summary of the whole sentence.
        encoder_output = encoder(input_ids, attention_mask)
        cls_token      = encoder_output[:, 0, :]    # (batch, hidden_dim)

        # ── Step 3: forward pass through our classification head ──
        logits = head(cls_token)                    # (batch, num_classes)

        # ── Step 4: compute loss ──
        # Compare our predictions (logits) to the correct labels
        loss = criterion(logits, labels)

        # ── Step 5: backward pass ──
        # PyTorch computes gradients for every parameter:
        # "how should I change each weight to reduce the loss?"
        loss.backward()

        # ── Step 6: update weights ──
        # The optimizer uses the gradients to nudge all weights
        # in the direction that reduces loss (gradient descent)
        optimizer.step()

        # Accumulate loss for reporting
        total_loss += loss.item()

    # Return average loss across all batches
    return total_loss / len(dataloader)


# ── PART 4: Inference with Threshold ─────────────────────────────────────────
#
# After training, we use the model to make predictions on new sentences.
#
# The key idea here is the CONFIDENCE THRESHOLD.
#
# Our model outputs logits → we apply sigmoid → we get probabilities (0 to 1).
#
# But we don't want to predict EVERY class — only the ones the model is
# confident about. So we set a threshold (e.g. 0.5):
#
#   sigmoid(logit) >= 0.5  →  predict this class  ✓
#   sigmoid(logit) <  0.5  →  skip this class     ✗
#
# A higher threshold (e.g. 0.8) = more conservative, fewer predictions, more precise
# A lower threshold  (e.g. 0.3) = more liberal, more predictions, more recall
#
# Example output for 3 sentences with 5 classes:
#   [[0, 2],    → sentence 0 belongs to classes 0 and 2
#    [1, 3, 4], → sentence 1 belongs to classes 1, 3, and 4
#    []]        → sentence 2: model not confident about any class

def predict_with_threshold(
    encoder:              nn.Module,
    head:                 MultiLabelClassificationHead,
    input_ids:            torch.Tensor,
    attention_mask:       torch.Tensor,
    confidence_threshold: float,
    device:               torch.device,
) -> list[list[int]]:
    """
    Runs inference and returns predicted class indices per sample.
    Only includes classes where sigmoid(logit) >= confidence_threshold.

    Args:
        encoder              : pretrained BERT-like encoder
        head                 : our trained classification head
        input_ids            : (batch_size, seq_len)
        attention_mask       : (batch_size, seq_len)
        confidence_threshold : float, e.g. 0.5
        device               : CPU or GPU

    Returns:
        List of lists — each inner list has the predicted class indices
        for that sentence. Empty list = no class was confident enough.
    """
    # Switch to eval mode — disables dropout, uses running stats for batch norm
    # IMPORTANT: always do this before inference, or results will be random
    encoder.eval()
    head.eval()

    # torch.no_grad() tells PyTorch: "don't track gradients"
    # We don't need gradients during inference (no backprop),
    # and skipping them saves memory and speeds things up.
    with torch.no_grad():

        # Move inputs to the correct device
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass through encoder → grab CLS token
        encoder_output = encoder(input_ids, attention_mask)
        cls_token      = encoder_output[:, 0, :]    # (batch, hidden_dim)

        # Forward pass through head → get logits
        logits = head(cls_token)                    # (batch, num_classes)

        # Apply sigmoid to convert logits → probabilities [0, 1]
        # Each value now represents "how confident are we about this class?"
        probabilities = torch.sigmoid(logits)       # (batch, num_classes)

    # Now apply the threshold — for each sentence, find which classes
    # exceeded the confidence threshold
    predictions = []

    for sample_probs in probabilities:
        # sample_probs is a 1D tensor of shape (num_classes,)
        # e.g. [0.12, 0.87, 0.03, 0.61, ...]

        # Find indices where probability >= threshold
        # .nonzero() returns the positions of True values
        predicted_indices = (sample_probs >= confidence_threshold).nonzero(as_tuple=True)[0]

        # Convert tensor of indices to a plain Python list of ints
        predictions.append(predicted_indices.tolist())

    return predictions


# ── Example Usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Multi-Label Classifier — Design Requirement Tagger")
    print("=" * 52)

    # Hyperparameters
    HIDDEN_DIM   = 768   # BERT's output size
    NUM_CLASSES  = 50    # number of verification categories
    THRESHOLD    = 0.5   # confidence threshold for prediction

    # Build the classification head
    head = MultiLabelClassificationHead(HIDDEN_DIM, NUM_CLASSES)

    # Count parameters
    total_params = sum(p.numel() for p in head.parameters())
    print(f"Classification head parameters: {total_params:,}")

    # Simulate a batch of 4 sentences (BERT output — random for demo)
    fake_cls_tokens = torch.randn(4, HIDDEN_DIM)

    # Forward pass
    logits = head(fake_cls_tokens)
    probs  = torch.sigmoid(logits)

    print(f"Input shape  : {fake_cls_tokens.shape}")
    print(f"Logits shape : {logits.shape}")
    print(f"Probs shape  : {probs.shape}")
    print(f"Sample probs (first sentence, first 10 classes):")
    print(f"  {probs[0, :10].detach().numpy().round(3)}")

    # Apply threshold
    for i, sample_probs in enumerate(probs):
        predicted = (sample_probs >= THRESHOLD).nonzero(as_tuple=True)[0].tolist()
        print(f"  Sentence {i}: predicted classes = {predicted}")
