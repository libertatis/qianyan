import collections
import json
import os
import random
import time
import numpy as np

import paddle

from squad_metrics import squad_evaluate


def prepare_train_features(examples, tokenizer, args):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    contexts = [examples[i]['context'] for i in range(len(examples))]
    questions = [examples[i]['question'] for i in range(len(examples))]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=args.doc_stride,
        max_seq_len=args.max_seq_length)

    for i, tokenized_example in enumerate(tokenized_examples):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_example["input_ids"]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offsets = tokenized_example['offset_mapping']

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        answers = examples[sample_index]['answers']
        answer_starts = examples[sample_index]['answer_starts']

        # If no answers are given, set the cls_index as answer.
        if len(answer_starts) == 0:
            tokenized_examples[i]["start_positions"] = cls_index
            tokenized_examples[i]["end_positions"] = cls_index
            tokenized_examples[i]['answerable_label'] = 0
        else:
            # Start/end character index of the answer in the text.
            start_char = answer_starts[0]
            end_char = start_char + len(answers[0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 2
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples[i]["start_positions"] = cls_index
                tokenized_examples[i]["end_positions"] = cls_index
                tokenized_examples[i]['answerable_label'] = 0
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[
                        token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples[i][
                    "start_positions"] = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples[i]["end_positions"] = token_end_index + 1
                tokenized_examples[i]['answerable_label'] = 1

    return tokenized_examples


def prepare_validation_features(examples, tokenizer, args):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    contexts = [examples[i]['context'] for i in range(len(examples))]
    questions = [examples[i]['question'] for i in range(len(examples))]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=args.doc_stride,
        max_seq_len=args.max_seq_length)

    # For validation, there is no need to compute start and end positions
    for i, tokenized_example in enumerate(tokenized_examples):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        tokenized_examples[i]["example_id"] = examples[sample_index]['id']

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples[i]["offset_mapping"] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_example["offset_mapping"])
        ]

    return tokenized_examples


def compute_prediction_checklist(examples,
                       features,
                       predictions,
                       version_2_with_negative: bool=False,
                       n_best_size: int=20,
                       max_answer_length: int=30,
                       cls_threshold: float=0.5):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).

            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
    """
    assert len(
        predictions
    ) == 3, "`predictions` should be a tuple with two elements (start_logits, end_logits, cls_logits)."
    all_start_logits, all_end_logits, all_cls_logits = predictions
    
    assert len(predictions[0]) == len(
        features), "Number of predictions should be equal to number of features."
    
    # Build a map example to its corresponding features.
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[feature["example_id"]].append(
            i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    all_cls_predictions = []

    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example['id']]

        min_null_prediction = None
        prelim_predictions = []
        score_answerable = -1
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            cls_logits = all_cls_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            exp_answerable_scores = np.exp(cls_logits - np.max(cls_logits))
            feature_answerable_score = exp_answerable_scores / exp_answerable_scores.sum()
            if feature_answerable_score[-1] > score_answerable:
                score_answerable = feature_answerable_score[-1]
                answerable_probs = feature_answerable_score
            if min_null_prediction is None or min_null_prediction[
                    "score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }
            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1:-n_best_size - 1:
                                                     -1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size - 1:-1].tolist(
            )
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (start_index >= len(offset_mapping) or
                            end_index >= len(offset_mapping) or
                            offset_mapping[start_index] is None or
                            offset_mapping[end_index] is None or
                            offset_mapping[start_index] == (0,0) or
                            offset_mapping[end_index] == (0,0)):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(
                            str(start_index), False):
                        continue
                    prelim_predictions.append({
                        "offsets": (offset_mapping[start_index][0],
                                    offset_mapping[end_index][1]),
                        "score":
                        start_logits[start_index] + end_logits[end_index],
                        "start_logit": start_logits[start_index],
                        "end_logit": end_logits[end_index],
                    })
        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            pred_cls_label = np.argmax(np.array(answerable_probs))
            all_cls_predictions.append([example['id'], pred_cls_label, answerable_probs[0], answerable_probs[1]])
        

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"],
            reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0)
                                               for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]:offsets[1]] if context[offsets[0]:offsets[1]] != "" else "no answer"

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and
                                     predictions[0]["text"] == "no answer"):
            predictions.insert(0, {
                "text": "no answer",
                "start_logit": 0.0,
                "end_logit": 0.0,
                "score": 0.0
            })

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            for n in range(len(predictions)):
                if predictions[i]["text"] != "no answer":
                    i = n
                    break
            # while predictions[i]["text"] == "no answer":
            #     i += 1
            best_non_null_pred = predictions[i]

            if answerable_probs[1] < cls_threshold:
                all_predictions[example['id']] = "no answer"
            else:
                all_predictions[example['id']] = best_non_null_pred['text']

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [{
            k: (float(v)
                if isinstance(v, (np.float16, np.float32, np.float64)) else v)
            for k, v in pred.items()
        } for pred in predictions]

    return all_predictions, all_nbest_json, all_cls_predictions


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, data_loader, args):
    model.eval()

    all_start_logits = []
    all_end_logits = []
    all_cls_logits = []
    tic_eval = time.time()

    for batch in data_loader:
        input_ids, segment_ids = batch
        start_logits_tensor, end_logits_tensor, cls_logits_tensor = model(input_ids, segment_ids)

        for idx in range(start_logits_tensor.shape[0]):
            if len(all_start_logits) % 1000 == 0 and len(all_start_logits):
                print("Processing example: %d" % len(all_start_logits))
                print('time per 1000:', time.time() - tic_eval)
                tic_eval = time.time()


            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])
            all_cls_logits.append(cls_logits_tensor.numpy()[idx])

    all_predictions, _, _ = compute_prediction_checklist(
        examples=data_loader.dataset.data, 
        features=data_loader.dataset.new_data,
        predictions=(all_start_logits, all_end_logits,all_cls_logits), 
        version_2_with_negative=True, 
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length, 
        cls_threshold=args.cls_threshold
    )

    exact_match, f1_score = squad_evaluate(
        examples=data_loader.dataset.data,
        preds=all_predictions,
        is_whitespace_splited=False
    )

    model.train()

    return exact_match, f1_score


@paddle.no_grad()
def predict(model, data_loader, args):
    model.eval()

    all_start_logits = []
    all_end_logits = []
    all_cls_logits = []
    tic_eval = time.time()

    for batch in data_loader:
        input_ids, segment_ids = batch
        start_logits_tensor, end_logits_tensor, cls_logits_tensor = model(input_ids, segment_ids)

        for idx in range(start_logits_tensor.shape[0]):
            if len(all_start_logits) % 1000 == 0 and len(all_start_logits):
                print("Processing example: %d" % len(all_start_logits))
                print('time per 1000:', time.time() - tic_eval)
                tic_eval = time.time()


            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])
            all_cls_logits.append(cls_logits_tensor.numpy()[idx])

    all_predictions, _, _ = compute_prediction_checklist(
        examples=data_loader.dataset.data, 
        features=data_loader.dataset.new_data,
        predictions=(all_start_logits, all_end_logits,all_cls_logits), 
        version_2_with_negative=True, 
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length, 
        cls_threshold=args.cls_threshold
    )

    return all_predictions


class CrossEntropyLossForChecklist(paddle.nn.Layer):
    def __init__(self):
        super(CrossEntropyLossForChecklist, self).__init__()

    def forward(self, y, label):
        start_logits, end_logits, cls_logits = y
        start_position, end_position, answerable_label = label

        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        answerable_label = paddle.unsqueeze(answerable_label, axis=-1)

        start_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=start_logits, label=start_position, soft_label=False)
        start_loss = paddle.mean(start_loss)

        end_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=end_logits, label=end_position, soft_label=False)
        end_loss = paddle.mean(end_loss)

        cls_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=cls_logits, label=answerable_label, soft_label=False)
        cls_loss = paddle.mean(cls_loss)

        mrc_loss = (start_loss + end_loss) / 2
        loss = (mrc_loss + cls_loss) /2

        return loss
