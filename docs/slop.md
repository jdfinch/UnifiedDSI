**Towards Streaming Inference of Task-Oriented Dialogue Structure**  

**Abstract**  
We introduce a novel, streaming-based approach to slot schema induction for task-oriented dialogue systems. Existing methods typically rely on clustering dense vector representations of slot-value candidates, which suffer from high sensitivity to hyperparameters, dependence on large datasets, and constraints related to semantic similarity. In contrast, our method frames schema induction as a text generation task, utilizing a streaming paradigm to incrementally construct schemas. This allows schemas to evolve in real time, maintaining adaptability across diverse dialogues.
To support our approach, we introduce the DOTS dataset—a fully automated, diverse, and schema-consistent corpus generated using GPT-4. Additionally, we propose new evaluation metrics that better align with human judgment, overcoming the limitations of previous metrics that relied on noisy embedding similarity-based matching.
We also demonstrate that typical evaluation benchmarks, such as MultiWOZ and SGD, are no longer suitable for testing in this domain because LLMs have memorized their slot schemas. To address this, we create a new evaluation dataset based on a novel set of domains. Our experiments, including comparisons with previous clustering-based models and large-scale benchmarks, show the effectiveness of our method in both synthetic and human-authored task scenarios.

---

**1. Introduction**  
Slot schema induction is a critical component of task-oriented dialogue systems. Traditional clustering-based methods group slot-value candidates based on semantic similarity, but these methods suffer from several limitations, including high sensitivity to hyperparameters and poor generalization to low-resource settings. We propose a novel streaming-based approach that formulates schema induction as a text generation problem. Our contributions include:
- A streaming induction model with mechanisms to manage schema growth.
- The DOTS dataset, which overcomes limitations of SGD and MultiWOZ.
- A new evaluation benchmark addressing LLM pre-memorization of existing slot schemas.
- New evaluation metrics based on boolean turn vectors and exact value match.

---

**2. Related Work**  
Previous work on slot schema induction relies on clustering methods that operate on dense vector representations of slot-value candidates. These methods require extensive hyperparameter tuning and large, diverse training datasets. Additionally, they depend on semantic similarity, which can lead to errors in schema generalization. Our work departs from clustering by leveraging text generation and streaming inference to dynamically induce schemas.

---

**3. Approach**  
We frame slot schema induction as a sequence modeling problem. Our approach sequentially generates schemas in a streaming manner, allowing the schema to evolve dynamically with new data. Key innovations include:
- **Quota Window Mechanism:** This prevents over-generation of slots by restricting schema updates within a fixed window.
- **Full-State vs. State-Update Prediction:** Full-state prediction maintains a holistic view of the dialogue state, whereas state-update prediction tracks incremental changes.
- **Dialogue-Level Induction:** New slots are only added at the end of a dialogue, enabling schema decisions based on the resolution of dialogue goals.

---

**4. The DOTS Dataset**  
Existing datasets (SGD, MultiWOZ, D0T) suffer from limited domain diversity or schema inconsistencies. DOTS was created through an automated pipeline using GPT-4o to generate:
1. Multi-domain task scenarios.
2. Schema representations as dataclasses.
3. Simulated dialogues following the generated schemas.
This results in a high-quality, schema-consistent dataset that enables effective training of our streaming induction model.

---

**5. Evaluation Setup**  
We critique traditional slot p/r/f1 and value p/r/f1 metrics and propose:
- **Turn-Vector Similarity Metric:** Matches predicted and gold slots based on turn-level boolean vectors.
- **Exact Value Match Metric:** Aligns closely with human judgments.

We evaluate on MultiWOZ 2.4 but note that LLMs have memorized its schema, trivializing evaluation. To counteract this, we introduce a new benchmark using DOTS-generated dialogues with human-authored scenarios and manual corrections.

---

**6. Results**  
We compare our approach against a baseline clustering model and observe:
- **DOTS outperforms D0T and SGD datasets for schema induction.**
- **Streaming schema induction (DSI) improves recall but initially struggles with precision.**
- **Quota windows mitigate the runaway schema problem.**
- **Dialogue-level induction enhances schema quality by leveraging full dialogue context.**
- **Our evaluation metrics better reflect human judgments compared to traditional metrics.**

---

**7. Conclusion and Future Work**  
Our streaming-based approach provides a more flexible and scalable method for slot schema induction compared to traditional clustering. We demonstrate the effectiveness of schema generation as a sequence modeling problem and introduce a robust evaluation framework. Future work will focus on refining our quota window mechanism, expanding DOTS, and applying our approach to assessment dialogues.
