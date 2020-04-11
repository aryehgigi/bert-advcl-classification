import torch
from torch.nn import BCELoss


class AdvclTransformer(torch.nn.Module):

    def __init__(self, base_model, class_weights=None):
        super(AdvclTransformer, self).__init__()
        if 'd_model' in vars(base_model.config):  # TODO (which is the correct for us)
            self.input_size = base_model.config.d_model
        else:
            self.input_size = base_model.config.hidden_size
        self.hidden_size = self.input_size  # TODO (use config)
        
        # added model  # TODO - revisit
        self.fc1 = torch.nn.Linear(self.input_size, int(self.hidden_size/2))
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(int(self.hidden_size/2), 1)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
        self.transformer = base_model
        self.class_weights = class_weights

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, args_indices=None):
        transformer_outputs = self.transformer(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output = transformer_outputs[0]

        batched_cat = []
        for i, (arg1_index, arg2_index, predicate_index) in enumerate(args_indices):
            batched_cat.append(output[i, [arg1_index, arg2_index, predicate_index]])
        output = torch.stack(batched_cat)

        logits = self.sigmoid(self.fc3(self.relu(self.fc2(self.relu(self.fc1(output)))).squeeze()[:,[0,1]])).squeeze()

        outputs = (logits,) + transformer_outputs[2:]
        
        if labels is not None:
            loss_fct = BCELoss(self.class_weights)
            loss = loss_fct(logits, labels)  # TODO
            outputs = (loss,) + outputs
        
        return outputs
