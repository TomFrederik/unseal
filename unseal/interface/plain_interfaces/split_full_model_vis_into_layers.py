import json
import argparse
import os

def main(args):
    with open(args.file, 'r') as f:
        data = json.load(f)

    # get model_name from path
    filename = os.path.basename(args.file)
    model_name = filename.split('.')[0]
    print(f"model_name: {model_name}")
    for i in range(2):
        if f'col_{i}' in data:
            for j in range(len(data[f'col_{i}'])):
                os.makedirs(f'{args.target_folder}/col_{i}', exist_ok=True)
                with open(f'{args.target_folder}/col_{i}/layer_{j}.txt', 'w') as f:
                    json.dump(data[f'col_{i}'][f'layer_{j}'], f)
        else:
            print(f"col_{i} not in data -> skipping")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--target_folder', type=str, required=True)
    args = parser.parse_args()
    
    main(args)
