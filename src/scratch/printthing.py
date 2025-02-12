

import dsi.dialogue as dial

dialogues = dial.dot2_to_dialogues('data/d0t/dot_2')
print(len(dialogues))

domains = {d for dialogue in dialogues for d in dialogue.domains()}
print(len(domains))

slots = {slot for dialogue in dialogues for slot in dialogue.schema}
print(len(slots))