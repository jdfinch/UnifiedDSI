

import ezpyzy as ez
import language_model.llama3 as llama
import dataclasses as dc


@dc.dataclass
class LinearDSI(ez.Config):
    model: llama.Llama3Config = llama.Llama3Config()

