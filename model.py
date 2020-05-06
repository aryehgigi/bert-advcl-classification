import torch
from torch.nn import CrossEntropyLoss, BCELoss


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
        self.fca2 = torch.nn.Linear(int(self.hidden_size), int(self.hidden_size/2))
        self.relua2 = torch.nn.ELU()
        self.fca3 = torch.nn.Linear(int(self.hidden_size/2), 1)
        self.sigmoida = torch.nn.Sigmoid()

        self.fcb1 = torch.nn.Linear(self.input_size, int(self.hidden_size / 2))
        self.relub = torch.nn.ELU()
        self.fcb2 = torch.nn.Linear(int(self.hidden_size), int(self.hidden_size/2))
        self.relub2 = torch.nn.ELU()
        self.fcb3 = torch.nn.Linear(int(self.hidden_size/2), 1)
        self.sigmoidb = torch.nn.Sigmoid()
        
        self.softmax = torch.nn.Softmax(dim=-1)
        
        self.transformer = base_model
        self.class_weights = class_weights

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, args_indices=None):
        transformer_outputs = self.transformer(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output = transformer_outputs[0]

        batched_cata = []
        batched_catb = []
        for i, (arg1_index, arg2_index, main_index, predicate_index) in enumerate(args_indices):
            batched_cata.append(output[i, [arg1_index, predicate_index]])
            batched_catb.append(output[i, [arg2_index, predicate_index]])
        outputa = torch.stack(batched_cata)
        outputb = torch.stack(batched_catb)

        out_a = self.sigmoida(self.fca3(self.relua2(self.fca2(self.relua(self.fca1(outputa)).view(-1, int(self.hidden_size)))))).squeeze()
        out_b = self.sigmoidb(self.fcb3(self.relub2(self.fcb2(self.relub(self.fcb1(outputb)).view(-1, int(self.hidden_size)))))).squeeze()
        combined = torch.stack([out_a, out_b], axis=-1)
        logits = self.softmax(combined)
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        

        outputs = (logits,) + transformer_outputs[2:]
        
        if labels is not None:
            labels = labels.type(torch.cuda.FloatTensor)
            loss_fct = CrossEntropyLoss(self.class_weights)
            loss = loss_fct(logits, labels.type(torch.cuda.LongTensor))
            outputs = (loss,) + outputs
        
        return outputs
