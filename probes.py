import json
import torch
import torch.nn as nn
from torch.nn import functional as F

class InterventionModule(nn.Module):
    '''
    input: padded hs_prime (original current step)
    output: padded hs (intervened step)
    '''
    def __init__(self, type, depth=1, input_size=4096, hidden_size=128, output_size=4096):
        super(InterventionModule, self).__init__()
        self.type = type
        self.depth = depth
        self.input_size = input_size
        self.output_size = output_size
        if self.type == 'lstm':
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.depth, batch_first=True).to('cuda')
            self.fc = nn.Linear(hidden_size, output_size).to('cuda')
            self.params = self.parameters()
        else:
            raise NotImplementedError("The probe type {type} is not defined!")

    def forward(self, input_h):
        if self.type == 'lstm': # not aggregate input, but aggregate output (I+O length, 1)
            input_h = input_h.to(torch.float32).to('cuda')
            h0 = torch.zeros(self.depth, input_h.size(0), self.hidden_size, dtype=torch.float32).to('cuda')
            c0 = torch.zeros(self.depth, input_h.size(0), self.hidden_size, dtype=torch.float32).to('cuda')
            # print(f'input_h.shape: {input_h.shape}') #逐渐递增 I -> I+O
            out, _ = self.lstm(input_h, (h0, c0))
            out = self.fc(out)

        return out

class Classifier():
    '''
    input: tensor(I+O length, hidden_size)
    output: bool (factual / not)
    '''
    type = '' #other types: nonlinear, lstm, bow
    # base_model = ''
    attn = None
    probe = None
    hidden_size = 4096
    mid_size = 2048 # non-linear only
    lstm_layer_num = 3 #lstm only
    output_size = 1 #binary classification by default
    params = None

    def __init__(self, type, hidden_size, output_size, aggregate_method='bow'):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.type = type
        self.aggregate_method = aggregate_method #only for linear / nonlinear probes, other method: max, last
        if self.aggregate_method == 'bow' and self.type != 'lstm': #Bag of Words
            self.attn = nn.Linear(self.hidden_size, 1)
        else:
            self.attn = torch.eye(self.hidden_size)
        if self.type == "linear":
            self.probe = nn.Sequential(
                nn.Linear(self.hidden_size, self.output_size),
                nn.Sigmoid(),
            )
        elif self.type == "nonlinear":
            self.probe = nn.Sequential(
                nn.Linear(self.hidden_size, self.mid_size),
                nn.ReLU(),
                nn.Linear(self.mid_size, self.output_size),
                nn.Sigmoid(),
            )
        elif self.type == "lstm":
            self.probe = nn.LSTM(self.hidden_size, self.output_size, num_layers=self.lstm_layer_num) #, bidirectionl=True
        else:
            raise NotImplementedError("The probe type {type} is not defined!")
        
        try:
            self.params = list(self.attn.parameters()) + list(self.probe.parameters()) #input for setting the optimizer
        except:
            self.params = list(self.probe.parameters()) #lstm has no self.attn

    def forward(self, input_h):
        self.probe.to('cuda')
        if self.type == 'lstm': # not aggregate input, but aggregate output (I+O length, 1)
            self.probe.flatten_parameters()
            input = input_h.view(input_h.shape[0], 1, -1).to(torch.float32).to('cuda') # reshape (I+O length, hidden_dim) to (I+O length, 1, hidden_dim)
            h_c = (
                torch.randn(self.lstm_layer_num, 1, self.output_size, dtype=torch.float32).to('cuda'), 
                torch.randn(self.lstm_layer_num, 1, self.output_size, dtype=torch.float32).to('cuda')
            ) #[2]Size(2, 1, 1)
            # print(f"Model device: {next(self.probe.parameters()).device}")
            # print(f"Input tensor device: {input.device}")
            # print(f"Label tensor device: {h_c[0].device}, {h_c[1].device}")
            out, h_c = self.probe(input, h_c) # #out: Size(I+O_length, 1, output_size), h_c: Size: [2](layer_num, 1, output_size)
            # print(f'out: {out}\nh_c.shape: {h_c}')
            assert torch.allclose(out[-1], h_c[0][-1]) #last output is the same as the last hidden state, both of shape tensor(1,2)
            sigmoid = nn.Sigmoid()
            output_h = sigmoid(h_c[0][-1]) #Size([1, 1]) #.reshape(1,2)
        else: #only consider aggregation when not using LSTM
            if self.aggregate_method == 'bow':
                # print(f"self.attn(input_h) shape: {self.attn(input_h).shape}") # torch.Size([191, 1])
                attn_weights = F.softmax(self.attn(input_h), dim=1) # normalize on the sequence length dim, # tensor(I+O length, 1)
                # print(f'attn weights shape: {attn_weights.shape}') # torch.Size([191, 1])
                h = attn_weights * input_h
                h = torch.sum(h, dim=0).unsqueeze(0)
                # print(f'h shape: {h.shape}') # torch.Size([1, 4096])
            elif self.aggregate_method == 'mean':
                h = torch.mean(input_h, dim=0).unsqueeze(0)
            elif self.aggregate_method == 'last':
                h = input_h[-1].unsqueeze(0)
            else:
                raise NotImplementedError("Aggregation method is not defined!")
        
            output_h = self.probe(h)

        return output_h.view(-1) #torch.Size([1]) #?
    
    def __call__(self, input_h):
        return self.forward(input_h)

    def to(self, device):
        for param in self.params:
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)

    def state_dict(self): #serializable
        attn_dict = {}
        probe_dict = {}
        if isinstance(self.attn, nn.Linear):
            for name, params in self.attn.named_parameters():
                attn_dict[name] = params.detach().cpu().numpy().tolist()
        for name, params in self.probe.named_parameters():
            probe_dict[name] = params.detach().cpu().numpy().tolist()

        if isinstance(self.attn, nn.Linear):
            state_dict = {
                'hidden_size': self.hidden_size,
                'mid_size': self.mid_size,
                'output_size': self.output_size,
                'type': self.type,
                'aggregate_method': self.aggregate_method,
                'attn': attn_dict,
                'probe': probe_dict,
            }
        else: #no attn dict for non-BoW classifiers
            state_dict = {
                'hidden_size': self.hidden_size,
                'mid_size': self.mid_size,
                'output_size': self.output_size,
                'type': self.type,
                'aggregate_method': self.aggregate_method,
                'probe': probe_dict,
            }
        return state_dict

    def load_state_dict(self, filepath):
        with open(filepath, 'r') as f:
            state_dict = json.load(f)

        # Load basic parameters
        self.hidden_size = state_dict['hidden_size']
        self.mid_size = state_dict['mid_size']
        self.output_size = state_dict['output_size']
        self.type = state_dict['type']
        self.aggregate_method = state_dict['aggregate_method']

        # Load weights for attention and probe if they exist
        if isinstance(self.attn, nn.Linear):
            for name, param in self.attn.named_parameters():
                param.data = torch.tensor(state_dict['attn'][name])
        
        for name, param in self.probe.named_parameters():
            param.data = torch.tensor(state_dict['probe'][name])
