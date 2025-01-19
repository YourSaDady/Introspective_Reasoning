'''
Iteratively train all kinds of classfiers / intervention modules for each loaded data split
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
from datasets import Dataset, load_dataset

sys.path.append('/h/382/momoka/HKU/honest_llama')
print(f'The current working directory: {os.getcwd()}')
from probes import Classifier, InterventionModule

# def load_HF_data_split(split_num): #TODO: load from remote HF hub, stil got errors
#     src_path = f'Lo-Fi-gahara/intervene_{split_num}k'
#     divs = {'train': 791, 'validation': 201}
#     train_set = []
#     validate_set = []
#     for div in divs:
#         print(f'Now loading on div: {div}...')
#         if div == 'train':
#             path_prefix = 'train/'
#         else:
#             path_prefix = 'validatation/'
#         for i in tqdm(range(divs[div])):
#             data_path = path_prefix + f'sample{i}/{div}-00000-of-00001.parquet'
#             dataset = load_dataset(src_path, split=div, data_dir=f'{path_prefix}sample{i}')
#             if div == 'train':
#                 train_set.append(dataset)
#             else:
#                 validate_set.append(dataset)
#             if i == 10: ##########
#                 break ##########

#     print(f'train_set[0]: {train_set[0]}, validate_set[0]: {validate_set[0]}')
#     return train_set, validate_set

'''限制intervention module输入的step hidden states长度为100'''
def pad(array, fix_len=256):
    if array.shape[0] > fix_len: #truncate the step length to be exa
        array = array[:fix_len, :]
    pad_len = fix_len - array.shape[0]
    try:
        paddings = torch.zeros((pad_len, array.shape[1]), dtype=torch.float32)
    except:
        print(f'\n!!!\nError: array has shape: {array.shape}, cannot be padded\n!!!\n')
        return torch.Tensor([-1]) 
    padded_array = torch.cat((array, paddings), axis=0)
    return padded_array

def build_hf_dataloader(hf_path, hf_batch_size=100, loader_batch_size=4, device='cuda', total_samples=500):
    all_h_h = [] #final step unit
    for b_start in  tqdm(range(0, total_samples, hf_batch_size)):
        dir_name = f'sample{b_start}-{b_start+hf_batch_size-1}'
        try:
            hf_batch_data = load_dataset(hf_path, split='train', data_dir=dir_name)
            for sample in hf_batch_data:
                new_hs_prime = [pad(torch.tensor(hs_prime, dtype=torch.float32)).to(device) for hs_prime in sample['h_prior']]
                new_hs = [pad(torch.tensor(hs, dtype=torch.float32)).to(device) for hs in sample['h_posterior']]
                all_h_h += [{'hs_prime': hs_prime, 'hs': hs} for (hs_prime, hs) in zip(new_hs_prime, new_hs)]
        #     try:
        #         assert len(all_hs) == len(all_hs_prime) 
        #     except AssertionError:
        #         print(f'Not matched length in hf_batch: "{dir_name}", with len(all_hs) = {len(all_hs)} and len(all_hs_prime) = {len(all_hs_prime)}')
        except:
            print(f'\nall hf batches are loaded! last sample: "sample{b_start}"\n')
    return DataLoader(all_h_h, batch_size=4, shuffle=False), len(all_h_h)
    
        


    print(f'\nloaded dataset with {len(dataset)} lines') #20
    print(f'\nsample0 ["h_prior"] length: \n___\n{len(dataset[0]["h_prior"])}\n___\n') #num_steps
    print(f'\nsample0 ["chosen_nodes"] length: \n___\n{len(dataset[0]["chosen_nodes"])}\n___\n')
    print(f'\nsample0 ["chosen_nodes"][0]: \n___\n{dataset[0]["chosen_nodes"][0]}\n___\n')

def build_data_loaders(train_path, validate_path, device, classify=False):
    train_set = []
    validate_set = []
    count = 0
    sample_count = 0
    empty_sample_count = 0
    pos_count = 0
    neg_count = 0
    for is_validate, path in enumerate([train_path, validate_path]):
        if is_validate: ############
            continue ############
        if is_validate: 
            print('Now loading validation set')
        else:
            print('Now loading training set')
        with open(path, 'r') as f:
            for line in tqdm(f): #iter samples
                count += 1
                # if count == 10: ############ for testing only
                #     break##########
                json_obj = json.loads(line)
                all_hs = list(json_obj.get('h_posterior'))
                all_hs_primes = list(json_obj.get('h_prior'))
                all_labels = list(json_obj.get('labels'))
                # print(f'len(all_hs_primes): {len(all_hs_primes)}') #show steps num
                assert len(all_hs_primes) == len(all_hs) == len(all_labels)
                for hs, hs_prime, label in zip(all_hs, all_hs_primes, all_labels): #iter steps
                    sample_count += 1
                    hs = pad(torch.tensor(hs, dtype=torch.float32)).to(device)
                    hs_prime = pad(torch.tensor(hs_prime, dtype=torch.float32)).to(device)
                    if torch.equal(hs, torch.Tensor([-1]).to(device)) or torch.equal(hs_prime, torch.Tensor([-1]).to(device)):
                        empty_sample_count += 1
                        continue
                    # hs = torch.tensor(hs, dtype=torch.float32).to(device)
                    # hs_prime = torch.tensor(hs_prime, dtype=torch.float32).to(device)

                    # label_bool = 1 if label == '+' else 0
                    # label = torch.tensor([label_bool], dtype=torch.float32).to(device) #batch_size = 1

                    if sample_count == 1: ############hs: torch.Size([100, 4096]), hs_prime: torch.Size([100, 4096]), label: '-'
                        print(f'hs: {hs}, hs_prime: {hs_prime}, label: {label}') ############
                        print(f'hs shape: {hs.shape}, hs_prime shape: {hs_prime.shape}, label: {label}') ############
                    if (classify and label == '+') or (not classify): #只保留positive classification label的samples用于训练
                        pos_count += 1
                        h_h = {'hs': hs, 'hs_prime': hs_prime} # torch.Size([100, 4096]), 1 or 0
                        if not is_validate: #train_set
                            train_set.append(h_h)
                        else:
                            validate_set.append(h_h)
                    elif classify and label == '-':
                        neg_count += 1
                        continue
    
    print(f'Loaded train set of size (pos: {len(train_set)}+{len(validate_set)} / total: {sample_count - empty_sample_count}) ({empty_sample_count} empty)')
    print(f'pos: {pos_count}, neg: {neg_count}')
    train_loader = DataLoader(train_set, batch_size=4, shuffle=False)
    validate_loader = DataLoader(validate_set, batch_size=4, shuffle=False)

    return train_loader, validate_loader, len(train_set), len(validate_set)



def h_m_s(start_t):
    spent = time.time() - start_t
    hrs = int(spent // 3600)
    spent %= 3600
    mins = int(spent // 60)
    secs = int(spent % 60)
    return f'{hrs}:{mins}:{secs}'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1-8b-instruct')
    parser.add_argument('--dataset_name', type=str, default='math-shepherd')
    parser.add_argument('--split_num', type=int, default=10, help='the number of dataset splits used. a single split contains 1k samples of the original dataset')
    parser.add_argument('--load_from_local', type=bool, default=False, help='source of the loaded data. Default to be False')
    parser.add_argument('--probe_type', type=str, default='classifier', help='Available probe types: [\'classifier\', \'intervention_module\']')
    parser.add_argument('--layer', type=int, default=16, help='the layer of the model to access the stat vars') #llama3.1-8b-instruct has 32 transformer layers, where the middle layers are supposed to be related to reasoning
    parser.add_argument('--positive_samples_only', type=bool, default=True, help='Valid when probe_type == "intervention_module". Whether to train on positive-labeled samples only')
    parser.add_argument('--test', type=bool, default=False, help='to be deleted (relate to save path)') ############
    args = parser.parse_args()

    '''
    1. Load dataset
    '''
    start_t = time.time()
    device = 'cuda'
    if args.probe_type == 'classifier' and args.load_from_local:
        print(f'\n***\nnow start loading classifier datasets with split ({args.split_num}) for layer ({args.layer})...\n***\n')
        train_path = f'./features/{args.model_name}_{args.layer}_{args.dataset_name}_{args.split_num}k_train_set.jsonl'
        validate_path = f'./features/{args.model_name}_{args.layer}_{args.dataset_name}_{args.split_num}k_validate_set.jsonl'
        train_set = []
        validate_set = []
        for is_validate, path in enumerate([train_path, validate_path]):
            # if not is_validate: ############
            #     continue ############
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
    # elif args.probe_type == 'intervention_module' and not args.load_from_local:
    #     print(f'\n***\nnow start loading classifier datasets with split ({args.split_num}) for layer ({args.layer}) from HF...\n***\n')
    #     train_set, validate_set = load_HF_data_split(args.split_num)
    elif args.probe_type == 'intervention_module' and args.load_from_local:
        print(f'\n***\nnow start loading intervener datasets with split ({args.split_num}) for layer ({args.layer}) from local...\n***\n')
        train_path = f'./features/{args.model_name}_{args.layer}_{args.dataset_name}_{args.split_num}k_train_set.jsonl'
        validate_path = f'./features/{args.model_name}_{args.layer}_{args.dataset_name}_{args.split_num}k_validation_set.jsonl'
        train_loader, validate_loader, train_size, validate_size = build_data_loaders(train_path, validate_path, device, classify=args.positive_samples_only)
    elif args.probe_type == 'intervention_module' and not args.load_from_local:
        if args.dataset_name == 'dpo':
            hf_path = f'Lo-Fi-gahara/intervener_layer{args.layer}_{args.split_num}k_dpo'
        train_loader, sample_size = build_hf_dataloader(hf_path, hf_batch_size=20)
        print(f'\nTotal samples: ({sample_size})\nTotal batches: ({len(train_loader)})\n')
    else:
        raise ValueError(f'Arguments are not implemented - probe_type: [{args.probe_type}] and load_from_local: [{args.load_from_local}]')
    print(f'Time spent on loading dataset: {h_m_s(start_t)}')

    # return


    '''
    2. Load probe
    '''
    all_probe_type = ['lstm', 'linear', 'nonlinear']
    all_aggregate_method = ['bow', 'mean']
    for type in all_probe_type:
        for aggregate in all_aggregate_method:
            if args.probe_type == 'intervention_module' and type == 'lstm':
                if args.test:
                    save_path = './trained_probes/interventor_lstm_test_.pth'
                    prior_path = save_path
                else:
                    save_path = f'./trained_probes/interventor_lstm_{args.split_num}k_classify-{args.positive_samples_only}.pth'
                    prior_path = f'./trained_probes/interventor_lstm_{args.split_num-1}k.pth'
            else:
                save_path = f'./trained_probes/{type}_{aggregate}_{args.model_name}_{args.layer}_{args.dataset_name}_{args.split_num}.json'
                prior_path = f'./trained_probes/{type}_{aggregate}__{args.model_name}_{args.layer}_{args.dataset_name}_{args.split_num}.json' #因为output size变了，所以重新训练
            print(f'\n---------------------\nnow start loading probe of type ({type}) and aggregate_method ({aggregate})...\n---------------------\n')
            if args.probe_type == 'classifier':
                probe = Classifier(
                    type,
                    hidden_size=4096,
                    output_size=1,
                    aggregate_method=aggregate
                )
            else:
                probe = InterventionModule(
                    type,
                    depth=1,
                )
            # load state_dict if there's any
            if os.path.exists(prior_path):
                state_dict = torch.load(prior_path)
                probe.load_state_dict(state_dict)

            probe.to(device)
            '''
            3. Train and validate probe
            '''
            start_t = time.time()
            #-------------------hyper params here---------------------
            # loss_func = nn.CrossEntropyLoss()
            if args.probe_type == 'classifier':
                loss_func = nn.BCELoss()
            else: #use MSE for continuous intervention
                loss_func = nn.MSELoss()
            optimizer = optim.AdamW(probe.params, lr=0.001) #default hyper params (lr = 0.001) 
            #---------------------------------------------------------
            # Prior training:
            if args.probe_type == 'classifer':
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
        
            # Start training:
            if args.probe_type == 'classifier':
                epoch = 1 #确保可以出现early stop
                for _ in range(epoch):
                    print(f'\n***\nnow start training classifier with epoch ({_})...\n***\n')
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
                                print(f'\nrunning_loss = {running_loss}, break!\n')
                                break
                            running_loss = 0.0
                    if 0.0 < running_loss < 0.04:
                        break
            else: #intervention module
                # print(f'- Total train size: {train_size}, validate size: {validate_size}') #classify的话显示positive samples数量
                epochs = 1 #确保可以出现early stop
                for _ in range(epochs):
                    print(f'\n***\nnow start training intervention module with epoch ({_})...\n***\n')
                    running_loss = 0.0
                    for i, hh_batch in enumerate(tqdm(train_loader)):
                        optimizer.zero_grad()
                        hs =  hh_batch['hs']
                        hs_prime = hh_batch['hs_prime']
                        output_logits = probe(hs_prime) #这里之前搞反了，需要重新训练下！
                        loss = loss_func(output_logits, hs)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item() #batch-wise
                        if i % 25 == 0: #每100 samples （25 batches）print一下
                        # if True: ##############
                        # if (i == 0 or i+1 == len(train_set)):    # print every 500 mini-batches
                            print(f'Epoch [{_+1}/{epochs}], Step [{i+1}/{len(train_loader)}], running_loss [{running_loss / (i+1):.5f}], loss [{loss.item():.5f}]')
                            # if running_loss < 0.04: #early stop
                            #     print(f'\running_loss = {running_loss}, break!\n')
                            #     break
                            # running_loss = 0.0
                    # if 0.0 < running_loss < 0.04:
                    #     break

            print('Finished Training')
            print(f'Time spent on loading dataset: {h_m_s(start_t)}')

            # Post training:
            if args.probe_type == 'classifier':
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
            if args.probe_type == 'classifier':
                with open(save_path, "w") as f:
                    json.dump(state_dict, f)
            elif args.probe_type == 'intervention_module' and type == 'lstm':
                torch.save(state_dict, save_path)
            print(f'trained probe saved to {save_path}, with split ({args.split_num})')

            if type == 'lstm':
                break

            break ####
        #aggregate ends
        break ####
    #probe_type ends

if __name__ == "__main__":
    main()
