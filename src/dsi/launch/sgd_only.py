
from __future__ import annotations

import random as rng
import sys
import pathlib as pl
import dataclasses as dc
import ezpyzy as ez
import copy as cp
import os

import dsi.experiment as ex
import dsi.data.pipelines as dp
import dsi.approach as app
import dsi.launch.launch as launch

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
    criterion_for_best_model=('valid_dst_sgd_resplit', 'mean_joint_goal_accuracy'),
    validate_every_n_steps=[100, 300, 500, 1000],
    validate_every_epoch=True,
    train_sgd_resplit=ex.TrainSGD_Resplit(),
    valid_dst_sgd_resplit=ex.ValidDST_SGD_Resplit(pipe=dp.DST_PerDomainEvaluationDataPipeline(
        downsample=dp.DownsampleDialogues(n=30)
    )),
    valid_dst_sgd_train_resplit=ex.ValidDST_SGD_TrainResplit(pipe=dp.DST_PerDomainEvaluationDataPipeline(
        downsample=dp.DownsampleDialogues(n=10)
    )),
    valid_dsi_mwoz=ex.ValidDSI_MWOZ(
        pipe=dp.DSI_EvaluationDataPipeline(
            downsample=dp.DownsampleDialogues(n=10)
        )
    ),
    eval_dst_sgd_resplit=ex.EvalDST_SGD_Resplit(),
    eval_dsi_mwoz=ex.EvalDSI_MWOZ(pipe=dp.DSI_EvaluationDataPipeline(
        downsample=dp.DownsampleDialogues(n=10)
    )),
    approach=app.LinearDSIConfig(
        model=llama.Llama3Config(
            model_base="meta-llama/Llama-3.2-1B-Instruct",
            adapter=lm.LoRA(rank=1),
            template_tokenizer=llama.Llama3TemplateTokenizerConfig(
                max_length=1024),
            generation=gen.Greedy(batch_size=5, num_kept_examples=3),
            training=lm.Training(
                optimizer=lm.Adam(learning_rate=1e-3, weight_decay=0),
                scheduler=lm.LinearWarmupSchedule(num_warmup_steps=100),
                batch_size=16,
                physical_batch_size=8,
                epochs=1,
                num_kept_examples=4
            )
        )
    )
)

print(experiment.configured.json())

# launch.launch(experiment)

def notes(): return
''' Notes



'''