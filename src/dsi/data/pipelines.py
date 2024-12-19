
import dataclasses as dc

from dsi.data.processors import *


@dc.dataclass
class DataProcessingPipeline(RandomProcess):
    load_path: str|tuple[str] = None
    rng_seed: int = None

    def __post_init__(self):
        super().__post_init__()
        self.data: ds.DSTData = None # noqa
        for name, processor in self.processors:
            if isinstance(processor, DataProcessor):
                processor.rng_seed = self.rng_seed

    @property
    def processors(self):
        for name, processor in self:
            if isinstance(processor, DataProcessor):
                yield name, processor

    def _set_rng_seed(self, rng_seed):
        super()._set_rng_seed(rng_seed)
        if self.configured.initialized:
            for name, processor in self.processors:
                if isinstance(processor, DataProcessor) and not processor.configured.has.rng_seed:
                    with processor.configured.not_configuring():
                        processor.rng_seed = rng_seed
        return rng_seed

    def process(self, data: ds.DSTData = None) -> ds.DSTData:
        if data is None:
            if isinstance(self.load_path, str):
                data = ds.DSTData(self.load_path)
            elif isinstance(self.load_path, (tuple, list)):
                datas = [ds.DSTData(path) for path in self.load_path]
                data = Concatenate().process(datas)
            else:
                raise ValueError(f"self.load_path must be str or tuple of str specifying path of data to load, got {self.load_path}")
        data = [cp.deepcopy(data)]
        for name, processor in self.processors:
            updated = []
            if isinstance(processor, Concatenate):
                updated.append(processor.process(data))
            else:
                for subdata in data:
                    processed_subdata = processor.process(subdata)
                    if isinstance(processed_subdata, list):
                        updated.extend(processed_subdata)
                    else:
                        updated.append(processed_subdata)
            data = updated
        processed, = data
        self.data = processed
        return processed


@dc.dataclass
class TrainingDataPipeline(DataProcessingPipeline):
    select_domains: SelectDomains|None = None
    downsample: DownsampleDialogues|None = None
    enable_multi_domain: EnableAllDomainsWithinEachDialogue|None = None
    fill_negatives: FillNegatives|None = FillNegatives()
    standardize_slot_names: StandardizeSlotNames|None = None

@dc.dataclass
class DST_EvaluationDataPipeline(DataProcessingPipeline):
    select_domains: SelectDomains|None = None
    downsample: DownsampleDialogues|None = None
    enable_multi_domain: EnableAllDomainsWithinEachDialogue|None = None
    fill_negatives: FillNegatives|None = FillNegatives()
    standardize_slot_names: StandardizeSlotNames|None = None

@dc.dataclass
class DST_PerDomainEvaluationDataPipeline(DataProcessingPipeline):
    select_domains: SelectDomains|None = None
    split_domains: SplitDomains|None = SplitDomains()
    downsample: DownsampleDialogues|None = None
    concatenate_domains: Concatenate|None = Concatenate()
    fill_negatives: FillNegatives|None = FillNegatives()
    standardize_slot_names: StandardizeSlotNames|None = None

@dc.dataclass
class DSI_EvaluationDataPipeline(DataProcessingPipeline):
    select_domains: SelectDomains|None = None
    downsample: DownsampleDialogues|None = None
    standardize_slot_names: StandardizeSlotNames|None = None



if __name__ == '__main__':
    ...