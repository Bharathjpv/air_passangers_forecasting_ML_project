import pickle as pkl
import matplotlib.pyplot as plt

def save_model(model, model_path):
    with open(model_path, 'wb') as md:
        pkl.dump(model, md)

def load_model(model_path):
    with open(model_path, 'rb') as md:
        load_name = pkl.load(md)

    return load_name
