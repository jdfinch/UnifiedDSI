
import dsi.dialogue as dial
import json

def get_schema(dialogues, running_schema, with_desc=False):
    for dialogue in dialogues:
        dialogue : dial.Dialogue
        for (domain, slot), (desc, _) in dialogue.schema.items():
            if domain not in running_schema:
                if not with_desc: 
                    running_schema[domain] = []
                else:
                    running_schema[domain] = {}

            if slot not in running_schema[domain]:
                if not with_desc:
                    running_schema[domain].append(slot)
                else:
                    running_schema[domain][slot] = desc


sgd_train = dial.sgd_to_dialogues(
    sgd_path='data/sgd/train',
    apply_sgdx=False,
    sgdx_rng_seed=None,
    remove_domain_numbers=True,
    filter_out_domains=[]
)

sgd_dev = dial.sgd_to_dialogues(
    sgd_path='data/sgd/dev',
    apply_sgdx=False,
    sgdx_rng_seed=None,
    remove_domain_numbers=True,
    filter_out_domains=[]
)

sgd_test = dial.sgd_to_dialogues(
    sgd_path='data/sgd/test',
    apply_sgdx=False,
    sgdx_rng_seed=None,
    remove_domain_numbers=True,
    filter_out_domains=[]
)

sgd_schema = {}
DESCRIPTION = True
get_schema(sgd_train, sgd_schema, with_desc=DESCRIPTION)
get_schema(sgd_dev, sgd_schema, with_desc=DESCRIPTION)
get_schema(sgd_test, sgd_schema, with_desc=DESCRIPTION)

print(f'Domains: {len(sgd_schema)}')
print(f'Slots: {len([slot for domain, slots in sgd_schema.items() for slot in slots])}')

print()
print(json.dumps(sgd_schema, indent=2))

...