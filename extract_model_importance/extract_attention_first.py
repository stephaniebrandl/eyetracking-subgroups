import numpy as np
import scipy.special

# This code has been dapted from bertviz: https://github.com/jessevig/bertviz/


def get_attention_for_sentence(model, tokenizer, sentence, modelname):
    # inputs = tokenizer.encode_plus(sentence, return_tensors='tf', add_special_tokens=True)
    # input_ids = inputs['input_ids']
    # attention = model(input_ids)[-1]

    input_ids = tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True).input_ids
    if modelname=='mt5':
        output = model(input_ids=input_ids, decoder_input_ids=input_ids, output_attentions=True)
        attention = output['encoder_attentions']
    else:
        output = model(input_ids=input_ids, output_attentions=True)
        attention = output['attentions']

    input_id_list = input_ids[0].numpy().tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)

    return tokens, attention


# For the attention baseline, we fixed several experimental choices (see below) which might affect the results.
def calculate_relative_attention( tokens, attention):
    # We use the last layer as Sood et al. 2020
    layer = 0

    # We use the first element of the batch because batch size is 1
    attention = attention[layer][0].detach().numpy()

    # 1. We take the mean over the 12 attention heads (like Abnar & Zuidema 2020)
    # I also tried the sum once, but the result was even worse
    mean_attention = np.mean(attention, axis=0)

    # We drop CLS and SEP tokens
    mean_attention = mean_attention[1:-1]

    # Optional: make plot to examine
    #    ax = sns.heatmap(mean_attention[1:-1, 1:-1], linewidth=0.5, xticklabels=tokens[1:-1], yticklabels=tokens[1:-1])
    #    plt.show()

    # 2. For each word, we sum over the attention to the other words to determine relative importance
    sum_attention = np.sum(mean_attention, axis=0)

    # Taking the softmax does not make a difference for calculating correlation
    # It can be useful to scale the salience signal to the same range as the human attention
    relative_attention = scipy.special.softmax(sum_attention)

    return tokens, relative_attention


def extract_attention_first(model, tokenizer, sentence, modelname):
    tokens, attention = get_attention_for_sentence(model, tokenizer, sentence, modelname)
    tokens, relative_attention = calculate_relative_attention(tokens, attention)

    return tokens, relative_attention
