import re


def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    text = re.sub(' +', ' ', text)
    text = text.strip()
    return text


def split_text_into_chunks(text, num_chunks):
    words = text.split()
    chunk_size = len(words) // num_chunks
    remainder = len(words) % num_chunks

    chunks_list = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        sentence = ' '.join(words[start:end])
        sentence = sentence.strip()
        chunks_list.append(sentence)
        start = end

    return chunks_list


def get_chunks_list(text, num_chunks):
    text_clean = clean_text(text)
    chunks_list = split_text_into_chunks(text_clean, num_chunks)
    return chunks_list