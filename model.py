import torch
from torch.nn import CrossEntropyLoss


class AdvclTransformer(torch.nn.Module):

    def __init__(self, base_model, class_weights=None):
        super(AdvclTransformer, self).__init__()
        if 'd_model' in vars(base_model.config):  # TODO (which is the correct for us)
            self.input_size = base_model.config.d_model
        else:
            self.input_size = base_model.config.hidden_size
        self.hidden_size = self.input_size  # TODO (use config)
        
        # added model  # TODO - revisit
        self.fca1 = torch.nn.Linear(self.input_size, int(self.hidden_size/2))
        self.relua = torch.nn.ELU()
        self.fca2 = torch.nn.Linear(int(self.hidden_size/2), 1)
        self.relua2 = torch.nn.ELU()

        self.fcb1 = torch.nn.Linear(self.input_size, int(self.hidden_size / 2))
        self.relub = torch.nn.ELU()
        self.fcb2 = torch.nn.Linear(int(self.hidden_size / 2), 1)
        self.relub2 = torch.nn.ELU()
        
        self.softmax = torch.nn.Softmax(dim=1)
        
        self.transformer = base_model
        self.class_weights = class_weights

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, args_indices=None):
        transformer_outputs = self.transformer(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output = transformer_outputs[0]

        batched_cata = []
        batched_catb = []
        for i, (arg1_index, arg2_index, predicate_index) in enumerate(args_indices):
            batched_cata.append(output[i, [arg1_index, predicate_index]])
            batched_catb.append(output[i, [arg2_index, predicate_index]])
        outputa = torch.stack(batched_cata)
        outputb = torch.stack(batched_catb)

        # logits = self.sigmoid(self.fc3(self.relu2(self.fc2(self.relu(self.fc1(output)))).squeeze()[:,[0,1]])).squeeze()
        out_a = self.relua2(self.fca2(self.relua(self.fca1(outputa))))
        out_b = self.relub2(self.fcb2(self.relub(self.fcb1(outputb))))
        combined = torch.stack([out_a, out_b], axis=1)
        logits = self.softmax(combined)[:,1].squeeze()

        outputs = (logits,) + transformer_outputs[2:]
        
        if labels is not None:
            labels = labels.type(torch.cuda.FloatTensor)
            loss_fct = CrossEntropyLoss(self.class_weights)
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs
        
        return outputs
