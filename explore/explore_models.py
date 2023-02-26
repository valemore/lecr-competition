from transformers import AutoModel, AutoTokenizer


TEXT = "Today yerk. two."

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased", use_fast=True)
foo = tokenizer(TEXT)
tokenizer.convert_ids_to_tokens(foo["input_ids"])

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
foo = tokenizer(TEXT)
tokenizer.convert_ids_to_tokens(foo["input_ids"])

tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", use_fast=True)
foo = tokenizer(TEXT)
tokenizer.convert_ids_to_tokens(foo["input_ids"])


special_tokens_dict = {'additional_special_tokens': ["[<<<]", "[TDSC]" , "[CDSC]", "[CTXT"]}
tokenizer.add_special_tokens(special_tokens_dict)

TEXT = "Title [<<<] One [<<<] Two [<<<] Root [TDSC] topic description"
foo = tokenizer(TEXT)
tokenizer.convert_ids_to_tokens(foo["input_ids"])

    additional_special_tokens = []
    for tok in (CROSS_ENT_START_TOKEN, CROSS_ENT_END_TOKEN, CROSS_ENT_SEP_TOKEN, CROSS_EXTRA_SEP_TOKEN):
        if tok not in tokenizer.get_vocab():
            additional_special_tokens.append(tok)
    if additional_special_tokens:
        special_tokens_dict = {'additional_special_tokens': [CROSS_ENT_SEP_TOKEN, CROSS_ENT_START_TOKEN,
                                                             CROSS_ENT_END_TOKEN, CROSS_EXTRA_SEP_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)
    num_new_tokens = len(additional_special_tokens)