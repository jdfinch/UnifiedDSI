

import random as rng
import sys
import pathlib as pl

import ezpyzy as ez
from language_model.llama import Llama

import dsi.data.structure as ds

import dataclasses as dc
from dataclasses import dataclass; vars().update(dataclass=ez.config) # noqa, black magic type hinting
import dsi.eval.metrics as metrics


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
        orginal_data = ds.DSTData(ExperimentConfig.train_data_path)


        model = Llama()
        all_domains = orginal_data.domains()
        #print(f"{model.generate('''What's the capital of France?''') = }")

        # self.train_data: ds.DSTData = ds.DSTData(self.train_data_path)

        for dialogue in orginal_data.dialogues.values():
            for turn in dialogue.turns:
                ...
    
        for epoch, ppl in enumerate(model.training(orginal_data)):
            prediction_data = copy.deepcopy(orginal_data)
            for slot_value in prediction_data.slots.values():
                slot_value = None # clear the gold
            for dialogue in orginal_data.dialogues:
                for turn in dialogue.turns:
                    all_domains.extend(turn.domains)
                    for slot, gold_value in turn.slot_values.items(): # check if the turn.slot_values.items() is valid
                        # naive version
                        prompt = format(dialogue, turn, slot.description)
                        predicted_value = model.generate(prompt)
                        prediction_data.slot_values[(dialogue.id,turn.index,turn.domains,slot.name)].value = predicted_value # check if turn_index is correct
                        #(dialogue_id, turn_index, domain, slot_name)
                        # real version
                        #... use a class to wrap Llama in a DST approach
            
            domain_jgas = {}
            for domain in all_domains:
                domain_jga = metrics.joint_goal_accuracy(domain, orginal_data, prediction_data)
                domain_jgas[domain] = domain_jga
                
            avg_jga = sum(domain_jgas.values()) / len(domain_jgas)
            
            results = metrics.EvaluationMetrics(domain_jgas, avg_jga)
            
            #model.save(f'ex/myexperiment/{epoch}/model')
            #ez.File('ex/myexperiment/{epoch}/metrics.json').save(dc.asdict(results))
            
            #ez.email(f"Epoch {epoch} complete.  Metrics: {metrics}")
         



if __name__ == '__main__':
    if len(sys.argv) > 1:
        experiment_path = sys.argv[1]
        Experiment(base=pl.Path(experiment_path)/'experiment.json')
        quit()

    Experiment(
        train_data_path='data/toy/train',
    )