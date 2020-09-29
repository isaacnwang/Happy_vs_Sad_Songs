import pickle as pkl
import numpy as np
from sklearn.ensemble import RandomForestClassifier


with open("final_model.pkl", "rb") as f:
    final_model = pkl.load(f)
f.close()


target_names=['happy','sad']
feature_names= ['danceability', 'energy', 'key', 
                      'loudness', 'mode', 'speechiness', 
                      'acousticness', 'instrumentalness', 'liveness', 
                      'valence','tempo']


def make_prediction(feature_dict):
    
    x_input = []

    for name in feature_names:
        x_input_ = float(feature_dict.get(name, 0))
        if x_input is None:
        	x_input=0
        x_input.append(x_input_)
        

    pred_probs = final_model.predict_proba([x_input]).flat
    probs = []
    for index in np.argsort(pred_probs)[::-1]:
        prob = {
            'name': target_names[index],
            'prob': round(pred_probs[index], 5)
        }
        probs.append(prob)
    return(x_input,probs)