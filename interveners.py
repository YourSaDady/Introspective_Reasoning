import torch
import torch.nn as nn
import pyvene as pv

def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

class Collector():
    collect_state = True
    collect_action = False  
    def __init__(self, head, multiplier, prefix_len=0):
        self.head = head
        self.states = []
        self.actions = []
        self.called_counter = 0
        self.prefix_len = prefix_len
        print(f'\ninside collector: self.prefix_len = {self.prefix_len}, self.head = {self.head}\n')
    def reset(self):
        self.states = []
        self.actions = []
        self.called_counter = 0
    def __call__(self, b, s, subspaces=None): 
        if subspaces['logging']:
            print(f"(called {self.called_counter} times) incoming reprs shape:", b.shape)
        self.called_counter += 1
        if self.head == -1:
            if self.called_counter == 1: #input hiddens
                self.states.append(b[:, self.prefix_len:, :].detach().clone()) #truncate the n-shot prefix
                # print(f'\nInside Collector: self.states[0].shape = {self.states[0].shape}, base.shape: {b.shape}\n')
            else:
                self.states.append(b.detach().clone())  #modified # original b is (batch_size, seq_len, #key_value_heads x D_head)
        else:
            self.states.append(b[0, -1].reshape(32, -1)[self.head].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        return b

'''
wrapper for the intervention Module
'''
class ClassifyIntervene(pv.ConstantSourceIntervention):
    def __init__(self, classifier, intervener, prefix_len, **kwargs):
        super().__init__(
            **kwargs,
            keep_last_dim=True, #to get token-level tokenized reprs
         )
        self.called_counter = 0
        self.classifier = classifier
        self.intervener = intervener
        self.prefix_len = prefix_len
        self.cache = [] # for storing past hidden states: max_new_tokens[torch.Size(1, n, 4096)], first n is full_len, otherwise = 1
        self.base_prefix = None

    def forward(self, base, source=None, subspaces=None): # core funcion
        #the first forward call returns the full length of input prompt and the first decoding token
        if self.cache == [] and base.size(1) > 1: 
            start_idx = self.prefix_len
            self.base_prefix = base[:, :start_idx, :].clone() #.unsqueeze(0) 
            base = base[:, start_idx:, :].clone() #exclude the prefix: torch.Size(1, n-prefix_len, 4096)
        if subspaces["logging"]: # self.prefix_len: 3436, base shape: torch.Size([1, 69, 4096])
            print(f"(called {self.called_counter+1} times) incoming reprs shape (after):", base.shape) #should be 100 times (torch.Size([1, 69, 4096]))
        self.called_counter += 1
        self.cache.append(base)
        cls_input = torch.cat([h.squeeze(0) for h in self.cache], dim=0) #torch.Size([prompt_len+decoding_len, 4096])
        itv_input = torch.cat([h[:, -1:, :] for h in self.cache], dim=0) #torch.Size([decoding_len, 4096])
        alpha = self.classifier(cls_input).item() # the classifier outputs a scalar, served as intervention strength (0~1)
        intervene_margin = self.intervener(itv_input)[-1:] #torch.Size([4096])
        new_base = base.clone()
        new_base[-1] = new_base[-1] + alpha * intervene_margin
        self.cache[-1] = new_base
        if self.called_counter == 1:
            print(f'\ninside first foward call: \n  - cls_input.shape: {cls_input.shape}\n  - itv_input.shape: {itv_input.shape}\n  - intervene_margin.shape: {intervene_margin.shape}')
            base = torch.cat((self.base_prefix, new_base), dim=1)
            print(f'intervened base shape: {base.shape} = {self.base_prefix.shape} + {new_base.shape}')
            return base
        else:
            return new_base

    
class ITI_Intervener():
    collect_state = True
    collect_action = True
    attr_idx = -1
    def __init__(self, direction, multiplier):
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction)
        self.direction = direction.cuda().half()
        self.multiplier = multiplier
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        self.states.append(b[0, -1].detach().clone())  # original b is (batch_size=1, seq_len, #head x D_head), now it's (#head x D_head)
        action = self.direction.to(b.device)
        self.actions.append(action.detach().clone())
        b[0, -1] = b[0, -1] + action * self.multiplier
        return b