import numpy as np

def conver_example(example, tokenizer, max_seq_length=512, is_test=False):
    query, title = example['sentence1'], example['sentence2']

    encoded_inputs = tokenizer(text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs['input_ids']
    token_type_ids = encoded_inputs['token_type_ids']

    if not is_test:
        label = np.array([example['label']], dtype='int64')
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids
