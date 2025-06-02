
import dsi.dialogue as dial

def get_data_stats(dialogues: dial.Dialogues):
    print('Number of dialogues:', n_dials:=len(dialogues))
    print('Number of turns:', n_turns:=sum(len(d.turns) for d in dialogues))
    print('Avg turns/dialogue:', n_turns/n_dials)
    print('Number of domains:', n_doms:=len({dom for d in dialogues for dom in d.domains()}))
    print('Number of slots:', n_slots:=len({s for d in dialogues for s in d.schema}))
    print('Avg slots/dom:', n_slots/n_doms)
    print('Number of value updates:', sum(len(list(d.updates())) for d in dialogues))


def compute_correction_stats(originals: dial.Dialogues, correcteds: dial.Dialogues):
    original_by_id = {}
    for dialogue in originals: original_by_id[dialogue.id] = dialogue
    corrected_by_id = {}
    for dialogue in correcteds: corrected_by_id[dialogue.id] = dialogue
    original_turns = set()
    corrected_turns = set()
    original_slot_values = set()
    corrected_slot_values = set()
    for dial_id, corrected in corrected_by_id.items():
        original = original_by_id[dial_id]
        for i, turn in enumerate(corrected.turns):
            corrected_turns.add((dial_id, i, turn))
        for i, turn in enumerate(original.turns):
            original_turns.add((dial_id, i, turn))
        for i, state in enumerate(corrected.states):
            for slot, value in state.items():
                corrected_slot_values.add((dial_id, i*2, slot, str(value)))
        for i, state in enumerate(original.states):
            for slot, value in state.items():
                original_slot_values.add((dial_id, i*2, slot, str(value)))
    original_untouched_turn_ids = {x[:2] for x in original_turns & corrected_turns}
    original_turns = {x[::2] for x in original_turns}
    corrected_turns = {x[::2] for x in corrected_turns}
    print()
    print('Turn Acc:', len(original_turns & corrected_turns)/len(original_turns))
    print('Turn IOU:', len(original_turns & corrected_turns)/len(original_turns | corrected_turns))
    print('Slot Value IOU:', len(original_slot_values & corrected_slot_values)/len(original_slot_values | corrected_slot_values))
    original_sv_uncorrected_turns = {x for x in original_slot_values if x[:2] in original_untouched_turn_ids}
    print('Percent Good Slot Values:', len(original_sv_uncorrected_turns & corrected_slot_values)/len(original_sv_uncorrected_turns))


'''
Stats of DOTS Eval Corrections:

Turn IOU: 0.7859820515733272
Turn Acc: 0.9278231987736331
Slot Value IOU: 0.8702420254050154
Percent Good Slot Values: 0.9569004196468632
'''    


if __name__ == '__main__':

    # mwoz = dial.multiwoz_to_dialogues('data/multiwoz24/train_dials.json')
    # print('\nMultiWOZ')
    # get_data_stats(mwoz)
    
    # sgd = dial.sgd_to_dialogues('data/sgd/train', apply_sgdx=False, filter_out_domains=())
    # print('\nSGD')
    # get_data_stats(sgd)

    sgd = dial.sgd_to_dialogues('data/sgd/test', apply_sgdx=False, filter_out_domains=())
    print('\nSGD Test')
    get_data_stats(sgd)
    print('SGD domains:', {d for x in sgd for d in x.domains()})

    # sgd = dial.sgd_to_dialogues('data/sgd/train', apply_sgdx=True, filter_out_domains=())
    # print('\nSGDX')
    # get_data_stats(sgd)

    # dot = dial.dot1_to_dialogues('data/d0t')
    # print('\nd0t')
    # get_data_stats(dot)

    # dots = dial.dot2_to_dialogues('data/DOTS/train')
    # print('\nDOTS')
    # get_data_stats(dots)

    eval_original = dial.dot2_to_dialogues('data/DOTS/eval')
    print('\nDOTS eval before corrections')
    get_data_stats(eval_original)
    
    eval_final = dial.dot2_to_dialogues('data/DOTS/eval_final_corrected')
    print('\nDOTS eval final')
    get_data_stats(eval_final)

    compute_correction_stats(eval_original, eval_final)

    # mwoz_eval = dial.multiwoz_to_dialogues('data/multiwoz24/test_dials.json')
    # print('\nMultiWOZ 2.4 test')
    # get_data_stats(mwoz_eval)

    '''
    MultiWOZ 2.4
    Number of dialogues: 8420
    Number of turns: 104916
    Avg turns/dialogue: 12.460332541567697
    Number of domains: 6
    Number of slots: 31
    Avg slots/dom: 5.166666666666667
    Number of value updates: 56668

    SGD
    Number of dialogues: 16142
    Number of turns: 329964
    Avg turns/dialogue: 20.44133316813282
    Number of domains: 16
    Number of slots: 115
    Avg slots/dom: 7.1875
    Number of value updates: 164982    

    SGDX
    Number of dialogues: 16142
    Number of turns: 329964
    Avg turns/dialogue: 20.44133316813282
    Number of domains: 16
    Number of slots: 763
    Avg slots/dom: 47.6875
    Number of value updates: 164982

    d0t
    Number of dialogues: 10002
    Number of turns: 195813
    Avg turns/dialogue: 19.577384523095382
    Number of domains: 1
    Number of slots: 173572
    Avg slots/dom: 173572.0
    Number of value updates: 100412

    DOTS
    Number of dialogues: 2771
    Number of turns: 88240
    Avg turns/dialogue: 31.844099603031395
    Number of domains: 787
    Number of slots: 6810
    Avg slots/dom: 8.653113087674715
    Number of value updates: 44120

    DOTS eval before corrections
    Number of dialogues: 1000
    Number of turns: 25716
    Avg turns/dialogue: 25.716
    Number of domains: 25
    Number of slots: 208
    Avg slots/dom: 8.32
    Number of value updates: 12858

    DOTS eval final
    Number of dialogues: 300
    Number of turns: 7844
    Avg turns/dialogue: 26.14666666666667
    Number of domains: 25
    Number of slots: 208
    Avg slots/dom: 8.32
    Number of value updates: 3922
    
    MultiWOZ 2.4 test
    Number of dialogues: 999
    Number of turns: 13737
    Avg turns/dialogue: 13.75075075075075
    Number of domains: 5
    Number of slots: 30
    Avg slots/dom: 6.0
    Number of value updates: 7368
    '''

