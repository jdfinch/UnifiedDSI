

import ezpyzy as ez
import dataclasses as dc
import language_model.llama3 as llama

import dsi.data.structure as ds



@dc.dataclass
class LinearDSIConfig(ez.Config):
    model: llama.Llama3Config = llama.Llama3Config()

class LinearDSI(ez.ImplementsConfig, LinearDSIConfig):

    def track(self, data: ds.DSTData):
        ...



if __name__ == '__main__':

    dsi = LinearDSIConfig()
    print(dsi.configured.json())

