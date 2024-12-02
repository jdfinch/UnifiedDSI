

import ezpyzy as ez
import language_model.llama3 as llama
import dataclasses as dc


@dc.dataclass
class LinearDSIConfig(ez.Config):
    model: llama.Llama3Config = llama.Llama3Config()

class LinearDSI(ez.ImplementsConfig, LinearDSIConfig):

    def track(self, _):
        ...



if __name__ == '__main__':

    dsi = LinearDSIConfig()
    print(dsi.configured.json())

