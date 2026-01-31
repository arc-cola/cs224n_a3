import json
import os
from pprint import pprint as pp

if __name__ == "__main__":
    results = []

    results_dir = './final/results/'
    for results_file_name in os.listdir(results_dir):
        with open(results_dir + results_file_name, 'r') as json_file:
            results.append(json.load(json_file))
    
    final_losses = [{'name': result['file_name'], 'loss': result['final_loss']} for result in results]
    pp(sorted(final_losses, key=lambda x: -x['loss']))