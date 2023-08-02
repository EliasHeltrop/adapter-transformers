import warnings

from ...model_mixin import EmbeddingAdaptersWrapperMixin
from ....file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ....models.vilt.modeling_vilt import VILT_INPUTS_DOCSTRING, VILT_START_DOCSTRING, ViltModel, ViltPreTrainedModel
from ...context import AdapterSetup
from ...heads import (
    ClassificationHead,
    ModelWithFlexibleHeadsAdaptersMixin,
    MultiLabelClassificationHead,
    MultipleChoiceHead,
    QuestionAnsweringHead,
)


@add_start_docstrings(
    """Vilt Model transformer with the option to add multiple flexible heads on top.""",
    VILT_START_DOCSTRING,
)
class ViltAdapterModel(EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.vilt = ViltModel(config)

        self._init_head_modules()

        self.init_weights()

    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head=None,
        **kwargs
    ):
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # BERT & RoBERTa return the pooled output as second item, we don't need that in these heads
        if not return_dict:
            head_inputs = (outputs[0],) + outputs[2:]
        else:
            head_inputs = outputs
        pooled_output = outputs[1]

        if head or AdapterSetup.get_context_head_setup() or self.active_head:
            head_outputs = self.forward_head(
                head_inputs,
                head_name=head,
                attention_mask=attention_mask,
                return_dict=return_dict,
                pooled_output=pooled_output,
                **kwargs,
            )
            return head_outputs
        else:
            # in case no head is used just return the output of the base model (including pooler output)
            return outputs

    head_types = {
        "classification": ClassificationHead,
        "multilabel_classification": MultiLabelClassificationHead,
        "multiple_choice": MultipleChoiceHead,
        "question_answering": QuestionAnsweringHead,
    }

    def add_classification_head(
        self,
        head_name,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        overwrite_ok=False,
        multilabel=False,
        id2label=None,
        use_pooler=False,
    ):
        """
        Adds a sequence classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
            multilabel (bool, optional): Enable multilabel classification setup. Defaults to False.
        """

        if multilabel:
            head = MultiLabelClassificationHead(
                self, head_name, num_labels, layers, activation_function, id2label, use_pooler
            )
        else:
            head = ClassificationHead(self, head_name, num_labels, layers, activation_function, id2label, use_pooler)
        self.add_prediction_head(head, overwrite_ok)

    def add_multiple_choice_head(
        self,
        head_name,
        num_choices=2,
        layers=2,
        activation_function="tanh",
        overwrite_ok=False,
        id2label=None,
        use_pooler=False,
    ):
        """
        Adds a multiple choice head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_choices (int, optional): Number of choices. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = MultipleChoiceHead(self, head_name, num_choices, layers, activation_function, id2label, use_pooler)
        self.add_prediction_head(head, overwrite_ok)


    def add_qa_head(
        self, head_name, num_labels=2, layers=1, activation_function="tanh", overwrite_ok=False, id2label=None
    ):
        head = QuestionAnsweringHead(self, head_name, num_labels, layers, activation_function, id2label)
        self.add_prediction_head(head, overwrite_ok)


class ViltModelWithHeads(ViltAdapterModel):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "This class has been renamed to `{}` in v3. "
            "Please use the new class instead as this class might be removed in a future version.".format(
                self.__class__.__bases__[0].__name__
            ),
            FutureWarning,
        )
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        warnings.warn(
            "This class has been renamed to `{}` in v3. "
            "Please use the new class instead as this class might be removed in a future version.".format(
                cls.__bases__[0].__name__
            ),
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        warnings.warn(
            "This class has been renamed to `{}` in v3. "
            "Please use the new class instead as this class might be removed in a future version.".format(
                cls.__bases__[0].__name__
            ),
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)