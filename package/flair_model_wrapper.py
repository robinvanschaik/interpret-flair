from flair.models import TextClassifier
from flair.data import Sentence
import torch
import torch.nn as nn
from transformers import AutoTokenizer

class ModelWrapper(nn.Module):

    def __init__(self, flair_model, layers: str = "-1"):
        super(ModelWrapper, self).__init__()

        # Pass the flair text classifier model object.
        # This is useful as we can inherit a lot of methods already made.
        self.flair_model = flair_model

        # Shorthand for the actual PyTorch model.
        self.model = flair_model.document_embeddings.model

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Split the name to automatically grab the right tokenizer.
        self.model_name = flair_model.document_embeddings.get_names()[0].split('transformer-document-')[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.label_dictionary = self.flair_model.label_dictionary
        self.num_classes = len(self.flair_model.label_dictionary)
        self.embedding_length = self.flair_model.document_embeddings.embedding_length

        self.initial_cls_token = flair_model.document_embeddings.initial_cls_token

        if layers == 'all':
            # send mini-token through to check how many layers the model has
            hidden_states = self.model(torch.tensor([1], device=device).unsqueeze(0))[-1]
            self.layer_indexes = [int(x) for x in range(len(hidden_states))]
        else:
            self.layer_indexes = [int(x) for x in layers.split(",")]

    def forward(self, input_ids):
        # Run the input embeddings through all the layers.
        # Return the hidden states of the model.
        hidden_states = self.model(input_ids=input_ids)[-1]

        # BERT has an initial CLS token.
        # Meaning that the the first token contains the classification.
        # Other models have this as the top layer.
        index_of_CLS_token = 0 if self.initial_cls_token else input_ids.shape[1] -1

        # For batching we need to replace
        # [layer][0][index_of_CLS_token]
        # with [layer][i][index_of_CLS_token].
        cls_embeddings_all_layers = \
            [hidden_states[layer][0][index_of_CLS_token] for layer in self.layer_indexes]

        output_embeddings = torch.cat(cls_embeddings_all_layers)

        # https://github.com/pytorch/captum/issues/355#issuecomment-619610044
        # It's better to attribute the logits to the inputs.
        label_scores = self.flair_model.decoder(output_embeddings)

        # Captum expects [#examples, #classes] as size.
        # We do to this so we can specify the target class with multiclass
        # models.
        label_scores_resized = torch.reshape(label_scores, (1, self.num_classes))
        return label_scores_resized
