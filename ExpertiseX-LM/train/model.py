import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput


class ExpertiseEmbeddings(nn.Module):
    """
    Extends standard RoBERTa embeddings by ADDING 6D expertise position features.
    Final embedding = word + position + token_type + e0 + e1 + e2 + e3 + e4 + e5
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 6D Expertise Position Embeddings (randomly initialized, learned during training)
        self.e0_embeddings = nn.Embedding(4, config.hidden_size)   # Node Role: Target=0, RevPaper=1, Area=2, None=3
        self.e1_embeddings = nn.Embedding(10, config.hidden_size)  # Graph Distance
        self.e2_embeddings = nn.Embedding(3, config.hidden_size)   # Relation to reviewer
        self.e3_embeddings = nn.Embedding(10, config.hidden_size)  # Field Depth
        self.e4_embeddings = nn.Embedding(3, config.hidden_size)   # Node Type
        self.e5_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)  # Token offset

        for emb in [self.e0_embeddings, self.e1_embeddings, self.e2_embeddings,
                     self.e3_embeddings, self.e4_embeddings, self.e5_embeddings]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False
        )
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

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

        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device)

        words_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeds + position_embeds + token_type_embeds

        if expertise_positions is not None:
            e0 = self.e0_embeddings(expertise_positions[:, :, 0])
            e1 = self.e1_embeddings(expertise_positions[:, :, 1])
            e2 = self.e2_embeddings(expertise_positions[:, :, 2])
            e3 = self.e3_embeddings(expertise_positions[:, :, 3])
            e4 = self.e4_embeddings(expertise_positions[:, :, 4])
            e5 = self.e5_embeddings(expertise_positions[:, :, 5])
            embeddings = embeddings + e0 + e1 + e2 + e3 + e4 + e5

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def _copy_pretrained_embeddings(custom_embeddings, pretrained_embeddings):
    """Copy pretrained word/position/token_type/LayerNorm weights into ExpertiseEmbeddings."""
    pretrained_vocab_size = pretrained_embeddings.word_embeddings.weight.size(0)
    custom_vocab_size = custom_embeddings.word_embeddings.weight.size(0)
    copy_size = min(pretrained_vocab_size, custom_vocab_size)
    custom_embeddings.word_embeddings.weight.data[:copy_size] = pretrained_embeddings.word_embeddings.weight.data[:copy_size]

    custom_embeddings.position_embeddings.weight.data.copy_(pretrained_embeddings.position_embeddings.weight.data)
    custom_embeddings.token_type_embeddings.weight.data.copy_(pretrained_embeddings.token_type_embeddings.weight.data)
    custom_embeddings.LayerNorm.weight.data.copy_(pretrained_embeddings.LayerNorm.weight.data)
    custom_embeddings.LayerNorm.bias.data.copy_(pretrained_embeddings.LayerNorm.bias.data)


class ExpertiseLMHead(nn.Module):
    """MLM head matching RoBERTa structure, supports weight tying with word_embeddings."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = self.decoder.bias

    def _tie_weights(self):
        self.bias = self.decoder.bias

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class ExpertiseXLMForPreTraining(RobertaPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.roberta.embeddings = ExpertiseEmbeddings(config)

        self.lm_head = ExpertiseLMHead(config)
        self.loss_fct = nn.CrossEntropyLoss()

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @classmethod
    def from_pretrained_phobert(cls, pretrained_name, config):
        """
        Initialize from PhoBERT pretrained weights:
        1. Load full PhoBERT backbone (encoder + embeddings)
        2. Copy encoder weights into our model
        3. Copy pretrained embedding weights into ExpertiseEmbeddings
        4. Copy LM head dense + layer_norm + decoder bias
        5. Decoder weight is tied to word_embeddings (auto-copied)
        6. Only the 6D expertise embeddings start from random init
        """
        model = cls(config)
        pretrained = RobertaForMaskedLM.from_pretrained(pretrained_name)

        # Copy encoder weights
        model.roberta.encoder.load_state_dict(pretrained.roberta.encoder.state_dict())

        # Copy pretrained embeddings (word_embeddings tied to decoder weight auto-updates)
        _copy_pretrained_embeddings(model.roberta.embeddings, pretrained.roberta.embeddings)

        # Copy LM head weights
        if hasattr(pretrained, 'lm_head'):
            pretrained_lm = pretrained.lm_head
            if hasattr(pretrained_lm, 'dense'):
                model.lm_head.dense.weight.data.copy_(pretrained_lm.dense.weight.data)
                model.lm_head.dense.bias.data.copy_(pretrained_lm.dense.bias.data)
            if hasattr(pretrained_lm, 'layer_norm'):
                model.lm_head.layer_norm.weight.data.copy_(pretrained_lm.layer_norm.weight.data)
                model.lm_head.layer_norm.bias.data.copy_(pretrained_lm.layer_norm.bias.data)
            if hasattr(pretrained_lm, 'decoder') and pretrained_lm.decoder.bias is not None:
                copy_size = min(pretrained_lm.decoder.bias.size(0), model.lm_head.decoder.bias.size(0))
                model.lm_head.decoder.bias.data[:copy_size] = pretrained_lm.decoder.bias.data[:copy_size]

        model.tie_weights()
        del pretrained

        print(f"[ExpertiseXLMForPreTraining] Loaded pretrained weights from '{pretrained_name}'")
        print(f"  - Encoder layers: copied")
        print(f"  - Word/Position/TokenType/LayerNorm embeddings: copied")
        print(f"  - LM Head (dense + layer_norm + decoder bias): copied")
        print(f"  - LM Head decoder weight: tied to word_embeddings")
        print(f"  - 6D Expertise embeddings (e0-e5): randomly initialized (std=0.02)")

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        expertise_positions=None,
        mlm_labels=None,
        **kwargs
    ):
        inputs_embeds = self.roberta.embeddings(input_ids=input_ids, expertise_positions=expertise_positions)

        outputs = self.roberta(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

        prediction_scores = self.lm_head(outputs[0])

        loss = None
        if mlm_labels is not None:
            loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ExpertiseXLMForSequenceClassification(RobertaPreTrainedModel):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.roberta.embeddings = ExpertiseEmbeddings(config)

        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.num_labels)
        )

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.post_init()

    @classmethod
    def from_pretrained_phobert(cls, pretrained_name, config):
        """Same as PreTraining but for classification (no LM head to copy)."""
        model = cls(config)
        pretrained = RobertaModel.from_pretrained(pretrained_name)

        model.roberta.encoder.load_state_dict(pretrained.encoder.state_dict())

        if pretrained.pooler is not None and model.roberta.pooler is not None:
            model.roberta.pooler.load_state_dict(pretrained.pooler.state_dict())

        _copy_pretrained_embeddings(model.roberta.embeddings, pretrained.embeddings)

        del pretrained

        print(f"[ExpertiseXLMForSequenceClassification] Loaded pretrained weights from '{pretrained_name}'")
        print(f"  - Encoder + Pooler: copied")
        print(f"  - Word/Position/TokenType/LayerNorm embeddings: copied")
        print(f"  - 6D Expertise embeddings (e0-e5): randomly initialized (std=0.02)")
        print(f"  - Classifier MLP: randomly initialized")

        return model

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

        logits = self.classifier(outputs[1])

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = self.bce_loss(logits.squeeze(-1), labels.view(-1).float())
            else:
                loss = self.ce_loss(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
