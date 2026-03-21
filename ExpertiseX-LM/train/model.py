import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput

class ExpertiseEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # 6 dimensions for expertise position embeddings
        self.e0_embeddings = nn.Embedding(4, config.hidden_size) # Node Role: Target=0, RevPaper=1, Area=2, None=3
        self.e1_embeddings = nn.Embedding(10, config.hidden_size) # Graph Distance
        self.e2_embeddings = nn.Embedding(3, config.hidden_size) # Relation to reviewer
        self.e3_embeddings = nn.Embedding(10, config.hidden_size) # Field Depth
        self.e4_embeddings = nn.Embedding(3, config.hidden_size) # Node Type
        self.e5_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size) # Token offset
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
        expertise_positions=None,
        **kwargs
    ):
        if inputs_embeds is not None:
            return inputs_embeds
            
        words_embeds = self.word_embeddings(input_ids)
        
        e0 = self.e0_embeddings(expertise_positions[:, :, 0])
        e1 = self.e1_embeddings(expertise_positions[:, :, 1])
        e2 = self.e2_embeddings(expertise_positions[:, :, 2])
        e3 = self.e3_embeddings(expertise_positions[:, :, 3])
        e4 = self.e4_embeddings(expertise_positions[:, :, 4])
        e5 = self.e5_embeddings(expertise_positions[:, :, 5])
        
        embeddings = words_embeds + e0 + e1 + e2 + e3 + e4 + e5
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class ExpertiseXLMForPreTraining(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # Override embeddings
        self.roberta.embeddings = ExpertiseEmbeddings(config)
        
        # MLM Head
        self.lm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.vocab_size)
        )
        
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        expertise_positions=None,
        mlm_labels=None,
        **kwargs # Catch other keys since collator might send classification_labels
    ):
        # We need to construct inputs_embeds ourself since RobertaModel doesn't accept expertise_positions
        inputs_embeds = self.roberta.embeddings(input_ids=input_ids, expertise_positions=expertise_positions)
        
        outputs = self.roberta(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        
        loss = None
        if mlm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            
        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ExpertiseXLMForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels # Using 1 for MSE since scores are float 1-5
        
        self.roberta = RobertaModel(config, add_pooling_layer=True)
        # Override embeddings
        self.roberta.embeddings = ExpertiseEmbeddings(config)
        
        # MLP Score predictor
        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.num_labels)
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        expertise_positions=None,
        labels=None,
        **kwargs
    ):
        inputs_embeds = self.roberta.embeddings(input_ids=input_ids, expertise_positions=expertise_positions)
        
        outputs = self.roberta(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Use BCE for binary match/no-match, MSE for explicit 1-5 scores
                is_binary = torch.all((labels == 0.0) | (labels == 1.0)).item()
                if is_binary:
                    loss_fct = nn.BCEWithLogitsLoss()
                else:
                    loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.view(-1).float())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
