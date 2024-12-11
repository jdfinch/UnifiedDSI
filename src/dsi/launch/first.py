
from __future__ import annotations

import random as rng
import sys
import pathlib as pl
import dataclasses as dc
import ezpyzy as ez
import copy as cp
import os

import dsi.data.structure as ds
import dsi.data.processing as dp
import dsi.eval.dst_eval as dst_eval
import dsi.eval.dsi_eval as dsi_eval
import dsi.approach as app
import dsi.launch.launch as launch
import dsi.experiment.experiment as ex

import language_model.generate as gen
import language_model.llama3 as llama
import language_model.tokens as tok
import language_model as lm


sgd_training_domains = [
    "Banks_1", "Buses_1", "Buses_2", "Calendar_1", "Events_1", "Events_2",
    "Flights_1", "Flights_2", "Homes_1",
    "Media_1", "Movies_1", "Music_1", "Music_2", "RentalCars_1", "RentalCars_2",
    "Services_1", "Services_2",
    "Services_3", "Weather_1"
]

sgd_testing_domains = [
    "Hotels_1", "Hotels_2", "Hotels_3", "Hotels_4",
    "Restaurants_1", "Restaurants_2",
    "RideSharing_1", "RideSharing_2",
    "Travel_1"
]


experiment = ex.ExperimentConfig(
    rng_seed=42,
    criterion_for_best_model=('dst_valid', 'slot_f1'),
    eval_every_n_steps=100,
    train_data_pipe=dp.DataProcessingPipeline(
        load_path='data/sgd/train',
        processors=ez.MultiConfig(
            domains=dp.SelectDomains(domains=sgd_training_domains, filter_dialogues=True),
            # downsample=dp.DownsampleDialogues(n=100),
            multi_domain=dp.EnableAllDomainsWithinEachDialogue(),
            standardize_slots=dp.StandardizeSlotNames(),
        )
    ),
    validations=ez.MultiConfig(
        dst_train=dst_eval.DST_Evaluation(
            pipe=dp.DataProcessingPipeline(
                load_path='data/sgd/train',
                processors=ez.MultiConfig(
                    domains=dp.SelectDomains(domains=sgd_training_domains, filter_dialogues=True),
                    downsample=dp.DownsampleDialogues(n=10),
                    standardize_slots=dp.StandardizeSlotNames(),
                )
            ),
            num_kept_examples=5,
        ),
        dst_valid=dst_eval.DST_Evaluation(
            pipe=dp.DataProcessingPipeline(
                load_path='data/sgd/valid',
                processors=ez.MultiConfig(
                    domains=dp.SelectDomains(domains=sgd_testing_domains, filter_dialogues=True),
                    downsample=dp.DownsampleDialogues(n=100),
                    standardize_slots=dp.StandardizeSlotNames(),
                )
            ),
            num_kept_examples=5,
        ),
    ),
    evaluations=ez.MultiConfig(
        discover=dsi_eval.DSI_Evaluation(
            pipe=dp.DataProcessingPipeline(
                load_path='data/sgd/valid',
                processors=ez.MultiConfig(
                    domains=dp.SelectDomains(domains=sgd_testing_domains, filter_dialogues=True),
                    downsample=dp.DownsampleDialogues(n=1),
                    standardize_slots=dp.StandardizeSlotNames(),
                )
            ),
            num_kept_examples=3,
        ),
    ),
    approach=app.LinearDSIConfig(
        model=llama.Llama3Config(
            model_base="meta-llama/Llama-3.1-8B-Instruct",
            template_tokenizer=llama.Llama3TemplateTokenizerConfig(
                max_length=1024
            ),
            adapters=ez.MultiConfig(adapter=lm.LoRA(
                rank=1, modules=('q_proj', 'v_proj', 'up_proj'),
            )),
            generation=gen.Greedy(batch_size=8, num_kept_examples=3),
            training=lm.Training(
                optimizer=lm.Adam(learning_rate=1e-3, weight_decay=1e-2),
                scheduler=lm.LinearWarmupSchedule(num_warmup_steps=100),
                batch_size=16,
                physical_batch_size=1,
                epochs=5,
                num_kept_examples=4
            )
        )
    )
)

print(experiment.configured.json())

launch.launch(experiment)

''' Notes

* try qwen

* The training is only registering values in the most recent domain even though all domains are included in the schema presented to the model. This contradicts the evaluation which properly includes all slot values of all schema slots, including domains that have been resolved earlier in the conversation.

* make sure slot discovery ALWAYS discovers ALL filled values that are not in the predefined DST schema 

'''