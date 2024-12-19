
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
    "Banks_1", "Buses_1", "Buses_2", "Calendar_1", "Events_1", "Events_2", "Events_3",
    "Flights_1", "Flights_2", "Flights_3", "Flights_4", "Homes_1",
    "Media_1", "Media_2", "Media_3", "Movies_1", "Music_1", "Music_2", "Music_3"
    "RentalCars_1", "RentalCars_2", "RentalCars_3"
    "Services_1", "Services_2", "Services_3", "Services_4", "Weather_1",
    "Alarm_1", "Messaging_1", "Payment_1"
]

sgd_testing_domains = [
    "Hotels_1", "Hotels_2", "Hotels_3", "Hotels_4",
    "Restaurants_1", "Restaurants_2",
    "RideSharing_1", "RideSharing_2",
    "Travel_1", # attractions
    "Trains_1"
]


experiment = ex.ExperimentConfig(
    rng_seed=42,
    criterion_for_best_model=('dst_valid', 'mean_joint_goal_accuracy'),
    validate_every_n_steps=100,
    train_data_pipe=dp.DataProcessingPipeline(
        load_path='data/d0t/train',
        processors=ez.MultiConfig(
            split_domains=dp.SplitDomains(),
            downsample=dp.DownsampleDialogues(n=1),
            cat_domains=dp.Concatenate(),
            multi_domain=dp.EnableAllDomainsWithinEachDialogue(),
            standardize_slots=dp.StandardizeSlotNames(add_domain_name=True),
        )
    ),
    validations=ez.MultiConfig(
        dst_valid=dst_eval.DST_PerDomainEvaluation(
            pipe=dp.DataProcessingPipeline(
                load_path='data/sgd/valid',
                processors=ez.MultiConfig(
                    domains=dp.SelectDomains(domains=sgd_testing_domains, filter_dialogues=True),
                    downsample=dp.DownsampleDialogues(n=100),
                    standardize_slots=dp.StandardizeSlotNames(add_domain_name=True),
                )
            ),
            num_kept_examples=10,
        ),
        discover=dsi_eval.DSI_Evaluation(
            pipe=dp.DataProcessingPipeline(
                load_path='data/sgd/valid',
                processors=ez.MultiConfig(
                    domains=dp.SelectDomains(domains=sgd_testing_domains, filter_dialogues=True),
                    downsample=dp.DownsampleDialogues(n=10),
                    standardize_slots=dp.StandardizeSlotNames(),
                )
            ),
            num_kept_examples=10,
        ),
    ),
    evaluations=ez.MultiConfig(
        dst_valid=dst_eval.DST_PerDomainEvaluation(
            pipe=dp.DataProcessingPipeline(
                load_path='data/sgd/valid',
                processors=ez.MultiConfig(
                    domains=dp.SelectDomains(domains=sgd_testing_domains, filter_dialogues=True),
                    standardize_slots=dp.StandardizeSlotNames(add_domain_name=True),
                )
            ),
            num_kept_examples=10,
        ),
        discover=dsi_eval.DSI_Evaluation(
            pipe=dp.DataProcessingPipeline(
                load_path='data/sgd/valid',
                processors=ez.MultiConfig(
                    domains=dp.SelectDomains(domains=sgd_testing_domains, filter_dialogues=True),
                    downsample=dp.DownsampleDialogues(n=10),
                    standardize_slots=dp.StandardizeSlotNames(),
                )
            ),
            num_kept_examples=10,
        ),
    ),
    approach=app.LinearDSIConfig(
        model=llama.Llama3Config(
            model_base="meta-llama/Llama-3.1-8B-Instruct",
            adapters=ez.MultiConfig(adapter=lm.LoRA(rank=1)),
            template_tokenizer=llama.Llama3TemplateTokenizerConfig(
                max_length=1024
            ),
            generation=gen.Greedy(batch_size=10, num_kept_examples=3),
            training=lm.Training(
                optimizer=lm.Adam(learning_rate=1e-3, weight_decay=0),
                scheduler=lm.LinearWarmupSchedule(num_warmup_steps=100),
                batch_size=64,
                physical_batch_size=1,
                epochs=1,
                num_kept_examples=30
            )
        )
    )
)

print(experiment.configured.json())

launch.launch(experiment)

def notes(): return
''' Notes

* try qwen

* set up option to save predictions and golds in each evaluation for analysis
    * splitting correct vs incorrect predictions
    
* save timings and usage stats

* add DSI validation

* visualize data
    * turn context
    * just turn
    * state or state update
    * dialogue
    * turn DST
    * turn DSI
    * schema matching
    * state comparison
    

'''