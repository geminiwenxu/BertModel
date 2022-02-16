from subwords.visualation import get_pacmap_pca_tsne_word_vs_x
from subwords.runner import this

if __name__ == "__main__":
    sentence_embeddings = this()
    embeddings = []
    for i in range(0, len(sentence_embeddings)):
        arr = sentence_embeddings[i].numpy()
        print(arr, len(arr))
        embeddings.append(arr)
    word_vec_list = [embeddings[0]]
    other_emb = [[embeddings[i]] for i in range(1, len(sentence_embeddings))]
    legend_names = [str(i) for i in range(0, len(sentence_embeddings))]
    output_dir = '/Users/wenxu/PycharmProjects/BertModel/subwords'
    name_title = 'this'
    get_pacmap_pca_tsne_word_vs_x(word_vec_list, other_emb, legend_names, output_dir, name_title)
