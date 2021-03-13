def accum_txt_sentences(chunk_len, txt):
    representation_way = []
    accum_sentence = []

    count = chunk_len
    for sentence in txt:
        accum_sentence.extend(sentence)
        if not count:
            representation_way.append(accum_sentence)
            accum_sentence = []
            count = 3
        count -= 1
    representation_way.append(accum_sentence)
    return representation_way
