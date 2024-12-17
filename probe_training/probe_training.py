'''
Iteratively train all kinds of probes for each loaded data split
'''

import time
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import argparse

sys.path.append('/h/382/momoka/HKU/honest_llama')
print(f'The current working directory: {os.getcwd()}')
from probes import Probe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1_8b_instruct')
    parser.add_argument('--dataset_name', type=str, default='math_shepherd')
    parser.add_argument('--split_num', type=int, default=1, help='the number of dataset splits used. a single split contains 1k samples of the original dataset')
    parser.add_argument('--layer', type=int, default=15, help='the layer of the model to access the stat vars') #llama3.1-8b-instruct has 32 transformer layers, where the middle layers are supposed to be related to reasoning
    args = parser.parse_args()

    '''
    1. Load dataset
    '''
    device = 'cuda'
    print(f'\n***\nnow start loading datasets with split ({args.split_num}) for layer ({args.layer})...\n***\n')
    train_path = f'./features/{args.model_name}_{args.layer}_{args.dataset_name}_{args.split_num}k_train_set.jsonl'
    validate_path = f'./features/{args.model_name}_{args.layer}_{args.dataset_name}_{args.split_num}k_validate_set.jsonl'
    train_set = []
    validate_set = []
    for is_validate, path in enumerate([train_path, validate_path]):
        if not is_validate: ############
            continue ############
        count = 0
        with open(path, 'r') as f:
            for line in tqdm(f):
                count += 1
                # if count == 10: ############ for testing only
                #     break##########
                json_obj = json.loads(line)
                hidden_states = torch.tensor(list(json_obj.get('hidden_states')), dtype=torch.float32).to(device)
                label = torch.tensor([int(json_obj.get('label'))], dtype=torch.float32).to(device) #batch_size = 1
                # if count == 1:
                #     print(f'loaded hidden states shape: {hidden_states.shape}, label: {label}')
                h_l = {'hidden_states': hidden_states, 'label': label} # torch.Size([100, 4096]), 1 or 0
                if not is_validate: #train_set
                    train_set.append(h_l)
                else:
                    validate_set.append(h_l)
    print(f'Loaded train set of size ({len(train_set)}) and validate set of size ({len(validate_set)})')

    '''
    2. Load probe
    '''
    all_probe_type = ['lstm', 'linear', 'nonlinear']
    all_aggregate_method = ['bow', 'mean']
    for type in all_probe_type:
        for aggregate in all_aggregate_method:
            if type == 'lstm':
                save_path = f'./trained_probes/layer3_lstm_{args.model_name}_{args.layer}_{args.dataset_name}_{args.split_num}_early_stop.json'
                prior_path = f'./trained_probes/layer3_lstm_{args.model_name}_{args.layer}_{args.dataset_name}_2k_best84.json' #因为output size变了，所以重新训练
            else:
                save_path = f'./trained_probes/{type}_{aggregate}_{args.model_name}_{args.layer}_{args.dataset_name}_{args.split_num}.json'
                prior_path = f'./trained_probes/{type}_{aggregate}__{args.model_name}_{args.layer}_{args.dataset_name}_{args.split_num}.json' #因为output size变了，所以重新训练
            print(f'\n---------------------\nnow start loading probe of type ({type}) and aggregate_method ({aggregate})...\n---------------------\n')
            probe = Probe(
                type,
                hidden_size=4096,
                output_size=1,
                aggregate_method=aggregate
            )
            # load state_dict if there's any
            if os.path.exists(prior_path):
                probe.load_state_dict(prior_path)

            probe.to(device)
            '''
            3. Train and validate probe
            '''
            #-------------------hyper params here---------------------
            # loss_func = nn.CrossEntropyLoss()
            loss_func = nn.BCELoss()
            optimizer = optim.AdamW(probe.params, lr=0.001) #default hyper params (lr = 0.001) 
            #---------------------------------------------------------
            # Prior training:
            total = len(validate_set)
            correct = 0
            with torch.no_grad():
                for i, h_l in enumerate(validate_set):
                    input_h =  h_l['hidden_states']
                    label = h_l['label']
                    output_logits = probe(input_h) #tensor([[0.7269, 0.2764]], device='cuda:0')
                    # pred = torch.argmax(output_logits, dim=1) # Size(1)
                    if i == 0:
                        print(f'output_logits.shape: {output_logits.shape}')
                        # print(f'output_logits.shape: {output_logits}, pred shape: {pred.shape}')
                    pred = 1 if output_logits.item() >= 0.5 else 0
                    if pred == label.item():
                        correct += 1
            prior_acc = 100 * correct // total
            print(f'Prior training, accuracy on validatation set of size ({total}): {prior_acc:.3f}%.')
        
            # return

            # Start training:
            epoch = 10 #确保可以出现early stop
            for _ in range(epoch):
                print(f'\n***\nnow start training probe with epoch ({_})...\n***\n')
                running_loss = 0.0
                for i, h_l in enumerate(tqdm(train_set)):
                    optimizer.zero_grad()
                    input_h =  h_l['hidden_states']
                    label = h_l['label']
                    output_logits = probe(input_h)
                    loss = loss_func(output_logits, label)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    if i % 500 == 499:
                    # if (i == 0 or i+1 == len(train_set)):    # print every 500 mini-batches
                        print(f'{i + 1:5d} running_loss: {running_loss / (i+1):.3f}, loss: {loss:.4f}')
                        if running_loss < 0.04: #early stop
                            print(f'\running_loss = {running_loss}, break!\n')
                            break
                        running_loss = 0.0
                if 0.0 < running_loss < 0.04:
                    break

            print('Finished Training')

            # Post training:
            correct = 0
            with torch.no_grad():
                for i, h_l in enumerate(validate_set):
                    input_h =  h_l['hidden_states']
                    label = h_l['label']
                    output_logits = probe(input_h) #tensor([[0.7269, 0.2764]], device='cuda:0')
                    pred = 1 if output_logits.item() >= 0.5 else 0
                    if pred == label.item():
                        correct += 1
            post_acc = 100 * correct // total
            print(f'Post training, accuracy on validatation set of size ({total}): {post_acc:.3f}%. (prior acc: {prior_acc})')

            '''
            4. Save probe
            '''
            state_dict = probe.state_dict()
            with open(save_path, "w") as f:
                json.dump(state_dict, f)
            print(f'trained probe saved to {save_path}, with split ({args.split_num})')

            if type == 'lstm':
                break

            break ####
        #aggregate ends
        break ####
    #probe_type ends

if __name__ == "__main__":
    main()
