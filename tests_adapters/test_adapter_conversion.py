import inspect
import re
import tempfile

import torch

from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    AlbertPreTrainedModel,
    AutoAdapterModel,
    BertPreTrainedModel,
    RobertaPreTrainedModel,
)
from transformers.testing_utils import require_torch, torch_device


@require_torch
class ModelClassConversionTestMixin:

    batch_size = 1
    seq_length = 128

    def run_test(self, static_model, input_shape=None, label_dict=None):
        flex_model = AutoAdapterModel.from_pretrained(None, config=self.config(), state_dict=static_model.state_dict())
        static_model.eval()
        flex_model.eval()
        if (
            static_model.base_model.__class__ != flex_model.base_model.__class__
            and not static_model.base_model == static_model
        ):
            self.skipTest("Skipping as base model classes are different.")

        with tempfile.TemporaryDirectory() as temp_dir:
            static_model.save_head(temp_dir)

            loading_info = {}
            flex_model.load_head(temp_dir, load_as="test", loading_info=loading_info)

        self.assertEqual(
            0, len(loading_info["missing_keys"]), "Missing keys: {}".format(", ".join(loading_info["missing_keys"]))
        )
        # We don't need to convert some of the weights, so remove them for the check
        unexpected_keys = loading_info["unexpected_keys"]
        if static_model._tied_weights_keys is not None:
            for pat in static_model._tied_weights_keys:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
        # HACK for bert-based models
        if isinstance(static_model, BertPreTrainedModel):
            unexpected_keys = [k for k in unexpected_keys if "cls.predictions.bias" not in k]
        elif isinstance(static_model, RobertaPreTrainedModel):
            unexpected_keys = [k for k in unexpected_keys if "lm_head.bias" not in k]
        elif isinstance(static_model, AlbertPreTrainedModel):
            unexpected_keys = [k for k in unexpected_keys if "predictions.bias" not in k]
        self.assertEqual(0, len(unexpected_keys), "Unexpected keys: {}".format(", ".join(unexpected_keys)))

        # adapter and head were loaded
        self.assertIn("test", flex_model.heads)

        # check equal output
        input_shape = input_shape or (self.batch_size, self.seq_length)
        in_data = self.get_input_samples(input_shape, config=flex_model.config)
        if label_dict:
            for k, v in label_dict.items():
                in_data[k] = v
        static_model.to(torch_device)
        flex_model.to(torch_device)
        output1 = static_model(**in_data)
        output2 = flex_model(**in_data)
        self.assertTrue(torch.allclose(output1.loss, output2.loss))
        self.assertTrue(torch.allclose(output1[1], output2[1]))  # it's not called "logits" for all classes

    def test_conversion_causal_lm_model(self):
        if self.config_class not in MODEL_FOR_CAUSAL_LM_MAPPING:
            self.skipTest("No causal language modeling class.")

        model = MODEL_FOR_CAUSAL_LM_MAPPING[self.config_class](self.config())
        label_dict = {}
        label_dict["labels"] = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        self.run_test(model, label_dict=label_dict)

    def test_conversion_masked_lm_model(self):
        if self.config_class not in MODEL_FOR_MASKED_LM_MAPPING:
            self.skipTest("No masked language modeling class.")

        model = MODEL_FOR_MASKED_LM_MAPPING[self.config_class](self.config())
        label_dict = {}
        label_dict["labels"] = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        # for encoder-decoder models such as BART, we additionally pass the decoder input ids
        if "decoder_input_ids" in inspect.signature(model.forward).parameters:
            label_dict["decoder_input_ids"] = label_dict["labels"].clone()
        self.run_test(model, label_dict=label_dict)

    def test_conversion_seq2seq_lm_model(self):
        if self.config_class not in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING:
            self.skipTest("No seq2seq language modeling class.")

        model = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING[self.config_class](self.config())
        label_dict = {}
        label_dict["labels"] = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        label_dict["decoder_input_ids"] = label_dict["labels"].clone()
        self.run_test(model, label_dict=label_dict)

    def test_conversion_classification_model(self):
        if self.config_class not in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING:
            self.skipTest("No sequence classification class.")

        model = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING[self.config_class](self.config())
        label_dict = {}
        label_dict["labels"] = torch.zeros(self.batch_size, dtype=torch.long, device=torch_device)
        self.run_test(model, label_dict=label_dict)

    def test_conversion_image_classification_model(self):
        if self.config_class not in MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING:
            self.skipTest("No image classification class.")

        model = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING[self.config_class](self.config())
        label_dict = {}
        label_dict["labels"] = torch.zeros(3, dtype=torch.long, device=torch_device)
        self.run_test(model, input_shape=(3, 3, 224, 224), label_dict=label_dict)

    def test_conversion_question_answering_model(self):
        if self.config_class not in MODEL_FOR_QUESTION_ANSWERING_MAPPING:
            self.skipTest("No question answering class.")

        model = MODEL_FOR_QUESTION_ANSWERING_MAPPING[self.config_class](self.config())
        label_dict = {}
        label_dict["start_positions"] = torch.zeros(self.batch_size, dtype=torch.long, device=torch_device)
        label_dict["end_positions"] = torch.zeros(self.batch_size, dtype=torch.long, device=torch_device)
        self.run_test(model, label_dict=label_dict)

    def test_conversion_token_classification_model(self):
        if self.config_class not in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING:
            self.skipTest("No token classification class.")

        model = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING[self.config_class](self.config())
        label_dict = {}
        label_dict["labels"] = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        self.run_test(model, label_dict=label_dict)

    def test_conversion_multiple_choice_model(self):
        if self.config_class not in MODEL_FOR_MULTIPLE_CHOICE_MAPPING:
            self.skipTest("No token classification class.")

        model = MODEL_FOR_MULTIPLE_CHOICE_MAPPING[self.config_class](self.config())
        label_dict = {}
        label_dict["labels"] = torch.ones(self.batch_size, dtype=torch.long, device=torch_device)
        self.run_test(model, input_shape=(self.batch_size, 2, self.seq_length), label_dict=label_dict)

    def test_equivalent_language_generation(self):
        if self.config_class not in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING:
            self.skipTest("no causal lm class.")

        static_model = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING[self.config_class](self.config())
        flex_model = AutoAdapterModel.from_pretrained(None, config=self.config(), state_dict=static_model.state_dict())
        static_model.add_adapter("dummy")
        static_model.set_active_adapters("dummy")
        static_model.eval()
        flex_model.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            static_model.save_adapter(temp_dir, "dummy")

            loading_info = {}
            flex_model.load_adapter(temp_dir, loading_info=loading_info)
            flex_model.set_active_adapters("dummy")

        input_shape = (self.batch_size, 5)
        input_samples = self.get_input_samples(input_shape, config=flex_model.config)

        static_model.to(torch_device)
        flex_model.to(torch_device)

        model_gen = static_model.generate(**input_samples)
        flex_model_gen = flex_model.generate(**input_samples)

        self.assertEquals(model_gen.shape, flex_model_gen.shape)
        self.assertTrue(torch.equal(model_gen, flex_model_gen))
