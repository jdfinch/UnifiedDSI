
from __future__ import annotations

import dsi.experiment as ex
import dsi.data.pipelines as dp
import dsi.approach as app
import dsi.launch.launch as launch

import language_model.generate as gen
import language_model.llama3 as llama
import language_model.tokens as tok
import language_model as lm


experiment = ex.ExperimentConfig(
    rng_seed=42,
    criterion_for_best_model=('valid_dst_sgd_resplit', 'mean_joint_goal_accuracy'),
    validate_every_n_steps=[100, 300, 500, 1000],
    validate_every_epoch=True,
    train_d0t=ex.TrainD0T(),
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

launch.launch(experiment)

def notes(): return
''' Notes



'''