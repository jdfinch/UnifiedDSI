
import dsi.dialogue as dial

def apply_fix(base_data, pred_data):
    source = dial.dot2_to_dialogues(base_data)
    idmap = {}
    for dialogue in source:
        content_sig = tuple(dialogue.turns)
        idmap[content_sig] = dialogue.id
    preds = dial.Dialogues.load(pred_data)
    for dialogue in preds:
        content_sig = tuple(dialogue.turns)
        dialogue.id = idmap[content_sig]
        print(dialogue.id)
    preds.save(pred_data)


if __name__ == '__main__':
    apply_fix(
        'data/DOTS/train',
        'ex/RKB_dc_noise_take2/0/dsi_dial_schemas.json'
    )
