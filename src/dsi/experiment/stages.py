
import dataclasses as dc

import dsi.data.pipelines as dp
import dsi.eval.dst_eval as dst_eval
import dsi.eval.dsi_eval as dsi_eval
import dsi.data.sgd.resplit as sgd_resplit


@dc.dataclass
class TrainSGD_Resplit(dp.TrainingDataPipeline):
    load_path: str = 'data/sgd/train'
    select_domains: dp.SelectDomains = dp.SelectDomains(domains=sgd_resplit.sgd_training_domains)
    enable_multi_domain: dp.EnableAllDomainsWithinEachDialogue = dp.EnableAllDomainsWithinEachDialogue()
    standardize_slot_names: dp.StandardizeSlotNames = dp.StandardizeSlotNames()


@dc.dataclass
class TrainD0T(dp.DataProcessingPipeline):
    load_path:str='data/d0t/train'
    downsample_dialogues: dp.DownsampleDialogues|None = None
    downsample_turns: dp.DownsampleTurns|None = None
    fill_negatives: dp.FillNegatives|None = dp.FillNegatives(max_negatives=5, max_negatives_factor=1.0)
    standardize_slot_names:dp.StandardizeSlotNames|None=dp.StandardizeSlotNames(add_domain_name=False)


@dc.dataclass
class ValidDST_SGD_TrainResplit(dst_eval.DST_PerDomainEvaluation):
    pipe: dp.DST_PerDomainEvaluationDataPipeline = dp.DST_PerDomainEvaluationDataPipeline(
        load_path='data/sgd/train',
        select_domains=dp.SelectDomains(domains=sgd_resplit.sgd_training_domains),
        standardize_slot_names=dp.StandardizeSlotNames(add_domain_name=True))
    num_kept_examples: int = 3
    use_for_training_validation: bool = True


@dc.dataclass
class ValidDST_SGD_Resplit(dst_eval.DST_PerDomainEvaluation):
    pipe: dp.DST_PerDomainEvaluationDataPipeline = dp.DST_PerDomainEvaluationDataPipeline(
        load_path='data/sgd/valid',
        select_domains=dp.SelectDomains(domains=sgd_resplit.sgd_testing_domains),
        standardize_slot_names=dp.StandardizeSlotNames(add_domain_name=True))
    num_kept_examples: int = 3
    use_for_training_validation: bool = True


@dc.dataclass
class ValidDST_MWOZ(dst_eval.DST_PerDomainEvaluation):
    pipe: dp.DST_PerDomainEvaluationDataPipeline = dp.DST_PerDomainEvaluationDataPipeline(
        load_path='data/multiwoz24/valid',
        select_domains=dp.SelectDomains(domains=['hotel', 'attraction', 'taxi', 'restaurant', 'train']),
        standardize_slot_names=dp.StandardizeSlotNames(add_domain_name=False))
    num_kept_examples: int = 3
    use_for_training_validation: bool = True


@dc.dataclass
class ValidDSI_MWOZ(dsi_eval.DSI_Evaluation):
    pipe: dp.DSI_EvaluationDataPipeline = dp.DSI_EvaluationDataPipeline(
        load_path='data/multiwoz24/valid',
        select_domains=dp.SelectDomains(domains=['hotel', 'attraction', 'taxi', 'restaurant', 'train']),
        standardize_slot_names=dp.StandardizeSlotNames(add_domain_name=False))
    num_kept_examples: int = 3
    use_for_training_validation: bool = True


@dc.dataclass
class EvalDST_SGD_Resplit(dst_eval.DST_PerDomainEvaluation):
    pipe: dp.DST_PerDomainEvaluationDataPipeline = dp.DST_PerDomainEvaluationDataPipeline(
        load_path='data/sgd/valid',
        select_domains=sgd_resplit.sgd_testing_domains,
        standardize_slot_names=dp.StandardizeSlotNames(add_domain_name=True))
    num_kept_examples: int = 3
    use_for_training_validation: bool = False


@dc.dataclass
class EvalDST_MWOZ(dst_eval.DST_PerDomainEvaluation):
    pipe: dp.DST_PerDomainEvaluationDataPipeline = dp.DST_PerDomainEvaluationDataPipeline(
        load_path='data/multiwoz24/valid',
        select_domains=dp.SelectDomains(domains=['hotel', 'attraction', 'taxi', 'restaurant', 'train']),
        standardize_slot_names=dp.StandardizeSlotNames(add_domain_name=False))
    num_kept_examples: int = 3
    use_for_training_validation: bool = False


@dc.dataclass
class EvalDSI_MWOZ(dsi_eval.DSI_Evaluation):
    pipe: dp.DSI_EvaluationDataPipeline = dp.DSI_EvaluationDataPipeline(
        load_path='data/multiwoz24/valid',
        select_domains=dp.SelectDomains(domains=['hotel', 'attraction', 'taxi', 'restaurant', 'train']),
        standardize_slot_names=dp.StandardizeSlotNames(add_domain_name=False))
    num_kept_examples: int = 3
    use_for_training_validation: bool = False




if __name__ == '__main__':

    print(TrainD0T().configured.json())
