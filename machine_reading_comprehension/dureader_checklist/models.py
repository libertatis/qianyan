import paddle
from paddle import nn
from paddlenlp.transformers import BertPretrainedModel
from paddlenlp.transformers import ErniePretrainedModel
from paddlenlp.transformers import RobertaPretrainedModel
from paddlenlp.transformers.model_utils import PretrainedModel


class ErnieGramPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained ERNIE-Gram models. It provides ERNIE-Gram related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. 
    Refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "ernie-gram-zh": {
            "attention_probs_dropout_prob": 0.1,
            "emb_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 18018
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "ernie-gram-zh":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_gram_zh/ernie_gram_zh.pdparams",
        },
    }
    base_model_prefix = "ernie_gram"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.ernie_gram.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-5

class ErnieGramForQuestionAnswering(ErnieGramPretrainedModel):
    """
    Model for Question and Answering task with ERNIE-Gram.
    Args:
        ernie_gram (`ErnieGramModel`): 
            An instance of `ErnieGramModel`.
    """

    def __init__(self, ernie_gram):
        super(ErnieGramForQuestionAnswering, self).__init__()
        self.ernie_gram = ernie_gram  # allow ernie_gram to be config
        self.classifier = nn.Linear(self.ernie_gram.config["hidden_size"], 2)
        self.classifier_cls = nn.Linear(self.ernie_gram.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
            token_type_ids (Tensor, optional):
                Segment token indices to indicate first and second portions of the inputs.
                Indices can be either 0 or 1:
                - 0 corresponds to a **sentence A** token,
                - 1 corresponds to a **sentence B** token.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
                Defaults to None, which means no segment embeddings is added to token embeddings.
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                config.max_position_embeddings - 1]``.
                Defaults to `None`. Shape as `(batch_sie, num_tokens)` and dtype as `int32` or `int64`.
            attention_mask (Tensor, optional):
                Mask to indicate whether to perform attention on each input token or not.
                The values should be either 0 or 1. The attention scores will be set
                to **-infinity** for any positions in the mask that are **0**, and will be
                **unchanged** for positions that are **1**.
                - **1** for tokens that are **not masked**,
                - **0** for tokens that are **masked**.
                It's data type should be `float32` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.
        Returns:
            A tuple of shape (``start_logits``, ``end_logits``).
            With the fields:
            - start_logits(Tensor): The logits of start position of prediction answer.
            - end_logits(Tensor): The logits of end position of prediction answer.
        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import ErnieGramForQuestionAnswering, ErnieGramTokenizer
                tokenizer = ErnieGramTokenizer.from_pretrained('ernie-gram-zh')
                model = ErnieGramForQuestionAnswering.from_pretrained('ernie-gram-zh')
                inputs = tokenizer("这是个测试样例")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
        """

        sequence_output, pooled_output = self.ernie_gram(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)
        cls_logits = self.classifier_cls(pooled_output)

        return start_logits, end_logits, cls_logits


class ErnieForQuestionAnswering(ErniePretrainedModel):
    def __init__(self, ernie):
        super(ErnieForQuestionAnswering, self).__init__()
        self.ernie = ernie  # allow ernie to be config
        self.classifier = nn.Linear(self.ernie.config["hidden_size"], 2)
        self.classifier_cls = nn.Linear(self.ernie.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, pooled_output = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)
        cls_logits = self.classifier_cls(pooled_output)

        return start_logits, end_logits, cls_logits

class BertForQuestionAnswering(BertPretrainedModel):
    def __init__(self, bert):
        super(BertForQuestionAnswering, self).__init__()
        self.bert = bert  # allow bert to be config
        self.classifier = nn.Linear(self.bert.config["hidden_size"], 2)
        self.classifier_cls = nn.Linear(self.bert.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, pooled_output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)
        cls_logits = self.classifier_cls(pooled_output)

        return start_logits, end_logits, cls_logits

class RobertaForQuestionAnswering(RobertaPretrainedModel):
    def __init__(self, roberta):
        super(RobertaForQuestionAnswering, self).__init__()
        self.roberta = roberta  # allow roberta to be config
        self.classifier = nn.Linear(self.roberta.config["hidden_size"], 2)
        self.classifier_cls = nn.Linear(self.roberta.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, pooled_output = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)
        cls_logits = self.classifier_cls(pooled_output)

        return start_logits, end_logits, cls_logits