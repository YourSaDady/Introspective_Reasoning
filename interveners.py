import torch
import torch.nn as nn
import pyvene as pv

def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

class Collector(pv.ConstantSourceIntervention):
    collect_state = True
    collect_action = False  
    def __init__(self, head, multiplier, prefix_len=0, **kwargs):
        super().__init__(
            **kwargs,
            keep_last_dim=True, #to get token-level tokenized reprs
         )
        self.head = head
        self.states = []
        self.actions = []
        self.called_counter = 0
        self.prefix_len = prefix_len
        # print(f'\ninside collector: self.prefix_len = {self.prefix_len}, self.head = {self.head}\n')
    def reset(self):
        self.states = []
        self.actions = []
        self.called_counter = 0
    def forward(self, b, s, subspaces=None): 
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

    def reset(self):
        print(f'您终于reset了。self.cache size: {len(self.cache)}')
        self.called_counter = 0
        self.cache.clear()
        self.base_prefix = None

    def forward(self, base, source=None, subspaces=None): # core funcion
        #the first forward call returns the full length of input prompt and the first decoding token
        self.called_counter += 1
        if self.called_counter == 1:
            self.base_prefix = base[:, :self.prefix_len, :] #.unsqueeze(0) 
            self.cache.append(self.base_prefix)
        else:
            self.cache.append(base)

        #TODO: 现在是每100tokens作为一个step进行一次clasify+intervene.完善的话可以单独训练一个预测step ending的classifier
        if self.called_counter == 100: #reached max new tokens
            self.called_counter = 0
            print(f'self.cache.length = {len(self.cache)}')

        cls_input = torch.cat([h.squeeze(0) for h in self.cache], dim=0) #torch.Size([prompt_len+decoding_len, 4096])
        alpha = self.classifier(cls_input).item() # the classifier outputs a scalar, served as intervention strength (0~1)

        if subspaces["logging"] and self.called_counter % 10 == 1: # self.prefix_len: 3436, base shape: torch.Size([1, 69, 4096])
            print(f"(called {self.called_counter} times) incoming reprs shape (after): {base.shape}, alpha: {alpha}") #should be 100 times (torch.Size([1, 69, 4096]))

        if alpha < 0.5: #intervene when negative
            itv_input = torch.cat([h[:, -1:, :] for h in self.cache], dim=1) #torch.Size([1, decoding_len, 4096])
            intervene_margin = self.intervener(itv_input)  #torch.Size([1, 618, 4096])
            intervene_margin = intervene_margin[-1, -1, :] #torch.Size([4096])
            new_base = base.clone()
            new_base[-1, -1] = intervene_margin #new_base[-1] - alpha * intervene_margin
            self.cache[-1] = new_base
            if subspaces["logging"]:
                alpha_after = self.classifier(torch.cat([h.squeeze(0) for h in self.cache], dim=0)).item()
                print(f'\n(called {self.called_counter} times) 您终于intervene了。\nalpha before: {alpha}, after: {alpha_after}\nbase before: {base.shape}, base after: {new_base.shape}\n')
            # self.called_counter = 0
            return new_base
        else:
            return base

    
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