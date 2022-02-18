from subwords.visualation import get_pacmap_pca_tsne_word_vs_x
from subwords.sentence_em import sentence_em

if __name__ == "__main__":
    sentence_embeddings, tokenized_text, common_tokens, diff_tokens = sentence_em()
    print(sentence_embeddings)
    print(tokenized_text)
    print(common_tokens)
    print(diff_tokens)
    wrong_token = []
    for i in diff_tokens:
        index = tokenized_text.index(i)
        token_em = sentence_embeddings[index]
        wrong_token.append(token_em)

    correct_token = []
    for i in common_tokens:
        index = tokenized_text.index(i)
        token_em = sentence_embeddings[index]
        correct_token.append(token_em)

    wrong = []
    for i in range(0, len(wrong_token)):
        arr = wrong_token[i].numpy()
        print(arr, len(arr))
        wrong.append(arr)
    word_vec_list = wrong
    correct = []
    for i in range(0, len(correct_token)):
        arr = correct_token[i].numpy()
        print(arr, len(arr))
        correct.append(arr)
    other_emb = [correct]
    print(word_vec_list)
    print(other_emb)
    legend_names = ['wrong', 'correct']
    output_dir = '/Users/wenxu/PycharmProjects/BertModel/subwords'
    name_title = 'this'
    get_pacmap_pca_tsne_word_vs_x(word_vec_list, other_emb, legend_names, output_dir, name_title)
