
# TACL 2025 


# Intro

## Task background

* Slot Schema Induction overview
    * discover slots from unlabeled dialogue data

### Impacts
* structural and problem analysis
* TOD development from H-H chat logs
* policy control from states

### Scientific Merits
* ontology creation/extension
* LLM ability to induce structure as text generation
    * traditionally, clustering is used

## Contributions

* novel approach to slot induction using LLM text generation, no clustering
    * take advantage of LLM intelligence for schema induction decisions
        * clustering relies on overly-compressed dense vectors representing slot values
        * no density-based sensitivity
        * can work with even 1 dialogue
    * allows a streaming approach
        * incremental online updates to adapt to changing data
        * minimal overhead to recompute slot schema
        * quality could scale gracefully as data grows
* DOTS training dataset with diverse schema-consistent dialogues
    * MultiWOZ and SGD have diversity limitations
    * d0t has a schema inconsistency issue
* comparison and validation of new evaluation metrics for schema induction
    * previous evaluation metrics are deeply flawed
* new DOTS evaluation dataset generated with human guidance and corrections
    * expands diversity of evaluation
    * MultiWOZ and SGD schemas are known to existing base models like GPT, Llama
* experiment results evaluate the impact of
    * LLM repeated discoveries as a metric of slot induction confidence
    * LLM ability to revise partial schemas
    * comparison to clustering methods
    * impact of dialogue data size on schema indcution
* publicly release models and code
    * DOTS train and corrected eval data
    * new state of the art schema induction model as llama-8B finetune


# Related Work

## Slot Schema Induction
... just iterate through them
* all are clustering based
* Finch et al 2024 current SoTA and uses LLM for sv candidate discovery
* previous evaluation metrics are flawed
    * precision doesn't punish redundant induced slots

## Evaluation Benchmark Leakage
* ?

## Data Generation

### for Symbolic Distillation
* Finch and Choi 2024
* ?

### for Evaluation Data
* ?


# Approach

* task formulation as citation
* overview summary

## Joint DST and SSE (Slot Schema Expansion)
* seq-to-seq formulation
* input: dialogue context, partial schema
* output: slot-values, new slot descriptions

variants:
* predict state updates, as in Finch et al 2024
* predict full states, similar to DST models
* predict full states only at the end of tasks
    * allows model to discover new slots with full task context
    * simulate task-end as the end of the dialogue in this work
        
## Schema Revision
* noisy slot discovery results in monotonically growing schema

### LLM-based schema revision
* inputs: predicted schema, dialogue example
* outputs: revised schema

variants:
* LLM revises schema
    * train on noisy schema predictions
* fifo with max threshold
* count slot occurences with a max threkshold
* count slot occurences within a window


# Data Generation
* mwoz and sgd have insufficient diversity for training
* d0t used in previous work but is schema-inconsistent
* overview
    * simulation-based data generation
    * schema -> dialogues instead of noisy annotation

## Scenario Creation
* scenarios support multi-domain task-oriented dialogues
* follow multiwoz's user-agent paradigm for finding/registering items
* diversity ensured by hand-checking scenarios
    * some domains share some semantic overlap
    * all scenario combinations are unique

## Schema Generation
* generate preferences dataclass given scenario description
    * scenario description partitioned into domains
    * each domain labeled with speaker tags and example slots
    * dataclass created to represent user preferences for each domain
* generate agent knowledge schema given scenario info and preference schema
    * subtle differences such as min_x for preference vs x for knowledge

## Task Initialization
* generate agent knoweledge
    * list of dataclass objects
* randomly select one object as the ideal user goal
* generate a user preferences object based on the ideal goal item
* augment the agent knowledge to make simulation non-trivial
    * generate red herring items similar to the goal item
    * some percent of the time, remove the ideal user goal from agent knowledge

## TOD Simulation
* generate dialogue alternating speaker turns
    * agent is conditioned on context, agent knowledge, and instructions
    * user is conditioned on context, user preferences, and instructions
* each user turn, annotate the dialogue state
    * dialogue state represents the current agent knowledge of user preferences
    * annotator uses preference schema dataclass and codes an object
* each agent turn, classify if the task is complete/incomplete/failed


# Evaluation Data
* all viable evaluation data is compromised via base model training
* evaluating on only weaker models also poses issues
    * weaker models may also have trained on benchmarks, but only results in latent biases rather than explicit recollection test success
    * observed findings may not generalize to strong models
* create a new dataset using human-corrected simulation
    * similar to SGD, but using LLMs instead of rules
    * not ideal, but provides an indication of performance on unseen domains

## Evaluation Data Leakage
* evaluation of multiwoz, sgd schema recollection
    * GPT-4o
    * Claude
    * Llama-8B

## Evaluation Data
* manual scenario authorship
    * intentionally focused on unique domains
* used data generation pipeline to generate schemas
* manually corrected schemas
    * slot removal, addition, and editing rates
* corrected schemas used to generate dialogues
* manually corrected 300 dialogues
    * turn edit and slot-value correction rates

## Dataset Stats
* comparison to multiwoz, sgd, d0t
* filtered out domains similar to multiwoz and test set
* correction rates of evaluation data are a proxy for training data quality


# Evaluation Metrics
* current standard evaluation metrics for slot schema induction are flawed
    * noisy automatic matching of predicted and gold slots
    * no punishment for redundant slot predictions
* analysis of agreement of automatic matchers with human matching
    * exact matcher
    * fuzzy matcher
    * embedding-based matcher
    * turn-occurence matcher


# Experiments

## Evaluations

Eval Data:
* DOTS eval set, per senario, macro-avgs
* multiwoz (all domains, as in previous work)

Train Data:
* DOTS train set
* SGD train set

Metrics:
* Slot P/R/F1
* Value P/R/F1


