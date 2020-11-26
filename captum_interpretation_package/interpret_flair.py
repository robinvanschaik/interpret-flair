import torch
import torch.nn as nn

from flair.data import Sentence

from captum.attr import (
    LayerIntegratedGradients
)

from captum.attr import visualization as viz


def interpret_sentence(flair_model_wrapper, lig, sentence, target_label, visualization_list, n_steps=100, estimation_method="gausslegendre",internal_batch_size=None):
    """
    We can visualise the attributions made by making use of Pytorch Captum.
    Inputs:
    flair_model_wrapper: class containing a customized forward function of Flair model.
    lig: the layer integrated gradient object.
    sentence: the Flair sentence-object we want to interpret.
    target_label: the ground truth class-label of the sentence.
    visualization_list: a list to store the visualization records in.
    """

    # Return the target index from the label dictionary.
    target_index = flair_model_wrapper.label_dictionary.get_idx_for_item(target_label)

    # In order maintain consistency with Flair, we apply the same tokenization
    # steps.
    flair_sentence = Sentence(sentence)

    tokenized_sentence = flair_sentence.to_tokenized_string()

    tokenizer_max_length = flair_model_wrapper.tokenizer.model_max_length

    # This calculates the token input IDs tensor for the model.
    input_ids = flair_model_wrapper.tokenizer.encode(tokenized_sentence,
                                                     add_special_tokens=False,
                                                     max_length=tokenizer_max_length,
                                                     truncation=True,
                                                     return_tensors="pt")

    # Create a baseline by creating a tensor of equal length
    # containing the padding token tensor id.
    pad_token_id = flair_model_wrapper.tokenizer.pad_token_id

    ref_base_line = torch.full_like(input_ids, pad_token_id)

    # Convert back to tokens as the model requires.
    # As some words might get split up. e.g. Caroll to Carol l.
    all_tokens = flair_model_wrapper.tokenizer.convert_ids_to_tokens(input_ids[0])

    # The tokenizer in the model adds a special character
    # in front of every sentence.
    readable_tokens = [token.replace("▁", "") for token in all_tokens]

    # The input IDs are passed to the embedding layer of the model.
    # It is better to return the logits for Captum.
    # https://github.com/pytorch/captum/issues/355#issuecomment-619610044
    # Thus we calculate the softmax afterwards.
    # For now, I take the first dimension and run this sentence, per sentence.
    model_outputs = flair_model_wrapper(input_ids)

    softmax = torch.nn.functional.softmax(model_outputs[0], dim=0)

    # Return the confidence and the class ID of the top predicted class.
    conf, idx = torch.max(softmax, 0)

    # Returns the probability.
    prediction_confidence = conf.item()

    # Returns the label name from the top prediction class.
    pred_label = flair_model_wrapper.label_dictionary.get_item_for_index(idx.item())

    # Calculate the attributions according to the LayerIntegratedGradients method.
    attributions_ig, delta = lig.attribute(input_ids,
                                           baselines=ref_base_line,
                                           n_steps=n_steps,
                                           return_convergence_delta=True,
                                           target=target_index,
                                           method=estimation_method,
                                           internal_batch_size=internal_batch_size)

    convergence_delta = abs(delta)
    print('pred: ', idx.item(), '(', '%.2f' % conf.item(), ')', ', delta: ', convergence_delta)


    word_attributions, attribution_score = summarize_attributions(attributions_ig)


    visualization_list.append(
    viz.VisualizationDataRecord(word_attributions=word_attributions,
                                pred_prob=prediction_confidence,
                                pred_class=pred_label,
                                true_class=target_label,
                                attr_class=target_label,
                                attr_score=attribution_score,
                                raw_input=readable_tokens,
                                convergence_score=delta)
                    )

    # Return these for the sanity checks.
    return readable_tokens, word_attributions, convergence_delta


def summarize_attributions(attributions):
    """
    Helper function for calculating word attributions.
    Inputs:
    attributions_ig: integrated gradients attributions.
    Ouputs:
    word_attributions: the attributions score per token.
    attribution_score: the attribution score of the entire document w.r.t. ground label.
    """
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attribution_score = attributions.sum()

    return attributions, attribution_score


def visualize_attributions(visualization_list):
    """
    Helper function to call Captum's visualization methods.
    Inputs:
    visualization_list: a list containing the integrated gradients attributions.
    """

    viz.visualize_text(visualization_list)
