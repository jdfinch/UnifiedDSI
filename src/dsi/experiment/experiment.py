

import random as rng
import sys
import pathlib as pl

import ezpyzy as ez
from language_model.llama import Llama

import dsi.data.structure as ds

import dataclasses as dc
from dataclasses import dataclass; vars().update(dataclass=ez.config) # noqa, black magic type hinting


@dataclass
class ExperimentConfig(ez.Config):
    name: str = ez.default(ez.denominate)
    description: str = ''
    path: str = None
    rng_seed: int = ez.default(rng.Random)
    train_data_path: str = None

    def __post_init__(self):
        super().__post_init__()
        if self.path is None: self.path = f'ex/{self.name}'


@dataclass
class Experiment(ExperimentConfig):
    def __post_init__(self):
        super().__post_init__()
        """
        Run the experiment! (V1)
        
        * process data  -  just load the toy data
        * train model -  (no training for now)
        * evaluate model - just print and look at it
        * save everything
        """
        data = ds.DSTData(ExperimentConfig.train_data_path)


        model = Llama()
        print(f"{model.generate('''What's the capital of France?''') = }")

        # self.train_data: ds.DSTData = ds.DSTData(self.train_data_path)

        """
    
        for epoch, ppl in enumerate(model.training(train_data)):
            prediction_data = copy.deepcopy(valid_data)
            for slot_value in prediction_data.slots.values():
                slot.value = None # clear the gold
            for dialogue in valid_data.dialogues:
                for turn in dialogue.turns:
                    for slot, gold_value in turn.slot_values.items():
                        # naive version
                        prompt = format(dialogue, turn, slot.description)
                        predicted_value = model.generate(prompt)
                        prediction_data.slot_values[figure_out_the_id].value = predicted_value
            
                        # real version
                        ... use a class to wrap Llama in a DST approach
            
            domain_jgas = {}
            for domain in valid_data.domains:
                domain_jga = joint_goal_accuracy(domain, valid_data, prediction_data)
                domain_jgas[domain] = domain_jga
                
            avg_jga = sum(domain_jgas.values()) / len(domain_jgas)
            
            metrics = EvaluationMetrics(domain_jgas, avg_jga)
            
            model.save(f'ex/myexperiment/{epoch}/model')
            ez.File('ex/myexperiment/{epoch}/metrics.json').save(dc.asdict(metrics))
            
            ez.email(f"Epoch {epoch} complete.  Metrics: {metrics}")
         
            
        """


if __name__ == '__main__':
    if len(sys.argv) > 1:
        experiment_path = sys.argv[1]
        Experiment(base=pl.Path(experiment_path)/'experiment.json')
        quit()

    Experiment(
        train_data_path='data/toy/train',
    )