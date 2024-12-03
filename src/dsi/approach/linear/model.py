

import ezpyzy as ez
import dataclasses as dc
import language_model.llama3 as llama

import dsi.data.structure as ds
import dsi.approach.linear.templates as temp



@dc.dataclass
class LinearDSIConfig(ez.Config):
    model: llama.Llama3Config = llama.Llama3Config(
        template_tokenizer=llama.Llama3TemplateTokenizerConfig(
            templates=temp.DSI_Templates()
        )
    )

class LinearDSI(ez.ImplementsConfig, LinearDSIConfig):

    def track(self, data: ds.DSTData):
        for dialogue in data:
            for turn in dialogue:
                if turn.schema is not None:
                    ...



if __name__ == '__main__':

    dsi = LinearDSIConfig()
    print(dsi.configured.json())

