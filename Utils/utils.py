import string
import random
import os
import numpy as np 

def load_session(load_dir, keywords):
    filenames = os.listdir(load_dir)
    matching_filenames = []
    for f in filenames:
        if np.all([k in f.split('_') for k in keywords]):
            matching_filenames.append(f)

    print("Number of matching filenames: %d\n"%len(matching_filenames), matching_filenames)
    

    matching_dicts = []
    for f in matching_filenames:
        d = np.load(load_dir+f, allow_pickle=True)
        matching_dicts.append(d)

    if len(matching_dicts) == 1:
        return matching_dicts[0].item()
    else:
        return matching_dicts

def save_session(save_dir, keywords, game_params, HPs, score, losses):
    ID = ''.join([random.choice(string.ascii_letters) for _ in range(4)])
    ID = ID.upper()
    keywords.append(ID)
    filename = '_'.join(keywords)
    filename = 'S_'+filename
    print("Save at "+save_dir+filename)
    train_session_dict = dict(game_params=game_params, HPs=HPs, score=score, n_epochs=len(score), keywords=keywords, losses=losses)
    np.save(save_dir+filename, train_session_dict)
    return ID
