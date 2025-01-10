

## Goal

Slot discovery with higher discovered slot consistency than clustering-based approaches.

## Motivation

* Dialogue state tracking already works well to consistently track a given slot
* LLMs with instruction can easily discover new, unique dialogue information types

## Approach

Train a joint dialogue state tracker and slot discovery model.

* Dialogue state tracking must be zero-shot to support newly discovered slots

## Formulation

Sequence:

1. Instruction
2. Existing slot definitions
3. Dialogue context
4. Tracked dialogue state
5. Discovered values
6. New slot definitions
