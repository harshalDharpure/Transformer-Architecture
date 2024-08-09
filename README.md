# Transformer-Architecture

# Transformer Architecture Overview

This repository provides a detailed theoretical overview of a small Transformer architecture. The Transformer model is a powerful neural network architecture widely used in Natural Language Processing (NLP), computer vision, and other sequence processing tasks.

## Table of Contents

- [Introduction](#introduction)
- [Components of Transformer Architecture](#components-of-transformer-architecture)
  - [1. Input Embedding Layer](#1-input-embedding-layer)
  - [2. Positional Encoding](#2-positional-encoding)
  - [3. Encoder Layer](#3-encoder-layer)
    - [a. Multi-Head Self-Attention](#a-multi-head-self-attention)
    - [b. Add & Norm (Residual Connection + Layer Normalization)](#b-add--norm-residual-connection--layer-normalization)
    - [c. Feed-Forward Neural Network](#c-feed-forward-neural-network)
  - [4. Transformer Encoder](#4-transformer-encoder)
- [Applications](#applications)
- [License](#license)

## Introduction

The Transformer model is a neural network architecture that has revolutionized various tasks involving sequence processing. Unlike traditional models that rely on recurrent structures, Transformers utilize self-attention mechanisms to capture relationships between tokens, enabling parallel processing of input sequences. This document provides an in-depth explanation of each component within a small Transformer architecture, illustrating their roles and functions.

## Components of Transformer Architecture

### 1. Input Embedding Layer

**Purpose:**
The Input Embedding Layer converts input tokens (such as words, subwords, or characters) into dense, continuous vector representations of fixed dimensions, often denoted as `d_model`.

**Function:**
- **Token Representation:** Each input token is mapped to a unique vector in a high-dimensional space, enabling the model to work with semantic information.
- **Dimensionality Reduction:** The embedding reduces the dimensionality of the input space, making it computationally feasible for the model to process.

**How It Helps:**
- **Semantic Representation:** By transforming discrete tokens into dense vectors, the model can capture semantic similarities between tokens, aiding in better understanding of the data.

### 2. Positional Encoding

**Purpose:**
Transformers lack inherent sequential order awareness because they process tokens in parallel. Positional Encoding introduces a way to inject order information, enabling the model to understand the position of each token in a sequence.

**Function:**
- **Position Information:** Positional encoding uses sine and cosine functions of varying frequencies to generate unique positional vectors, which are added to the token embeddings.
- **Frequency Variation:** The model can distinguish between different positions in long sequences, understanding relative positions between tokens.

**How It Helps:**
- **Order Awareness:** Positional encoding ensures that the model considers the position of each token in the sequence, crucial for tasks where word order impacts meaning.

### 3. Encoder Layer

Each Encoder Layer in the Transformer model comprises several key components:

#### a. Multi-Head Self-Attention

**Purpose:**
The Multi-Head Self-Attention mechanism allows the model to focus on different parts of the input sequence simultaneously, capturing complex relationships between tokens.

**Function:**
- **Attention Scores:** Computes attention scores between every pair of tokens in the sequence, determining the importance of other tokens when processing a given token.
- **Multi-Head Mechanism:** Multiple attention heads allow the model to capture different aspects of relationships (e.g., syntactic, semantic) between tokens in parallel.

**How It Helps:**
- **Contextual Understanding:** Enables the model to build a rich, contextual understanding of each token by considering its relationships with all other tokens in the sequence.

#### b. Add & Norm (Residual Connection + Layer Normalization)

**Purpose:**
The Add & Norm sub-layer stabilizes the learning process by ensuring smooth gradient flow and maintaining consistent output scales.

**Function:**
- **Residual Connection:** Adds the output of the self-attention or feed-forward network to the input of the sub-layer, preserving information from earlier layers.
- **Layer Normalization:** Normalizes the summed result to maintain a stable distribution of activations, improving training stability.

**How It Helps:**
- **Training Stability:** Residual connections aid in gradient flow, while normalization speeds up training and avoids issues like vanishing or exploding gradients.

#### c. Feed-Forward Neural Network

**Purpose:**
The Feed-Forward Neural Network introduces non-linearity, allowing the model to learn complex transformations of the input features.

**Function:**
- **Layer Composition:** Consists of two linear transformations with a ReLU activation in between, processing each position independently.
- **Independent Processing:** The FFN processes each position independently, without considering interactions between positions (handled by self-attention).

**How It Helps:**
- **Complex Feature Transformation:** The non-linear transformations enable the model to capture intricate patterns in the data, essential for tasks like natural language understanding.

### 4. Transformer Encoder

**Purpose:**
The Transformer Encoder is a stack of multiple encoder layers that transforms the input sequence into a rich, contextually aware representation, useful for tasks such as classification, translation, or sequence generation.

**Function:**
- **Stacked Layers:** Composed of multiple identical layers, each containing Multi-Head Self-Attention, Add & Norm, and Feed-Forward Neural Network components.
- **Hierarchical Representation:** The input passes through each layer, progressively refining its representation to capture increasingly complex patterns and dependencies.

**How It Helps:**
- **Comprehensive Understanding:** The stacked layers enable the model to build a hierarchical understanding of the input, capturing both local and global dependencies between tokens.

## Applications

Transformers are used in a wide range of applications, including but not limited to:
- **Natural Language Processing (NLP):** Tasks like translation, summarization, sentiment analysis, etc.
- **Vision Transformers:** Adapted for computer vision tasks where the model processes image patches as sequences.
- **Speech Processing:** Applied in tasks like speech recognition and synthesis.


