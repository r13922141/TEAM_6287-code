import os
import json
import argparse
import numpy as np
from tqdm import tqdm


def flag_retrieve_without_reranker():
    import time
    import joblib
    from sentence_transformers import SentenceTransformer
    start = time.perf_counter()
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(
        description='Process some paths and files.')
    parser.add_argument('--question_path', type=str,
                        required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--output_path', type=str,
                        required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案
    # read and load finance dataset embedding vector
    with open("embedded_dataset/finance6.pickle", 'rb') as f:
        corpus_dict_finance = joblib.load(f)
    # read and load insurance dataset embedding vector
    with open("embedded_dataset/insurance.pickle", 'rb') as f:
        corpus_dict_insurance = joblib.load(f)
    # read and load faq dataset embedding vector
    with open("embedded_dataset/faq.pickle", 'rb') as f:
        corpus_dict_faq = joblib.load(f)
    # use SentenceTransformer for loading embedding model BAAI/bge-m3，can download the model by execute the code
    model = SentenceTransformer("BAAI/bge-m3")
    for q_dict in tqdm(qs_ref['questions']):
        qs = q_dict['query']
        source = q_dict['source']

        if q_dict['category'] == 'finance':
            """Process finance query"""

            filter_corpus = [corpus_dict_finance[int(
                file)][0] for file in q_dict['source']]
            embedded_corpus = []
            idp = []
            k = 0
            #紀錄每一段資料的source id
            for x in filter_corpus:
                embedded_corpus.extend(x)
                tmp = [k]*len(x)
                idp.extend(tmp)
                k += 1
            # convert query to embedding vector and run in GPU
            embedded_query = model.encode(qs, convert_to_tensor=True)
            embedded_query = embedded_query.cpu().numpy()
            similarity = [np.dot(embedded_query, corpus)
                          for corpus in embedded_corpus]

            # 由大排到小
            scoremax = np.argmax(similarity)

            retrieved = source[idp[scoremax]]

            # 將結果加入字典
            answer_dict['answers'].append(
                {"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            """
            Process insurance query
            Same process with finance
            """

            filter_corpus = [corpus_dict_insurance[int(
                file)][0] for file in q_dict['source']]
            embedded_corpus = []
            idp = []
            k = 0
            #紀錄每一段資料的source id
            for x in filter_corpus:
                embedded_corpus.extend(x)
                tmp = [k]*len(x)
                idp.extend(tmp)
                k += 1
            # convert query to embedding vector and run in GPU
            embedded_query = model.encode(qs, convert_to_tensor=True)
            embedded_query = embedded_query.cpu().numpy()
            similarity = [np.dot(embedded_query, corpus)
                          for corpus in embedded_corpus]

            scoremax = np.argmax(similarity)  # 由大排到小

            retrieved = source[idp[scoremax]]

            answer_dict['answers'].append(
                {"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            """
            Process insurance query
            Same process with finance
            """
            filter_corpus = [
                corpus_dict_faq[int(file)][0] for file in q_dict['source']]
            embedded_corpus = []
            idp = []
            k = 0
            #紀錄每一段資料的source id
            for x in filter_corpus:
                embedded_corpus.extend(x)
                tmp = [k]*len(x)
                idp.extend(tmp)
                k += 1
            # convert query to embedding vector and run in GPU
            embedded_query = model.encode(qs, convert_to_tensor=True)
            embedded_query = embedded_query.cpu().numpy()
            similarity = [np.dot(embedded_query, corpus)
                          for corpus in embedded_corpus]

            scoremax = np.argmax(similarity)  # 由大排到小

            retrieved = source[idp[scoremax]]

            answer_dict['answers'].append(
                {"qid": q_dict['qid'], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False,
                  indent=4)  # 儲存檔案，確保格式和非ASCII字符
    end = time.perf_counter()
    print("total time:", end-start)


def flag_retrieve_with_reranker():
    """

    Retrieve most similar source according to query

    """
    import time
    import joblib
    from sentence_transformers import SentenceTransformer
    from FlagEmbedding import LayerWiseFlagLLMReranker
    start = time.perf_counter()
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(
        description='Process some paths and files.')
    parser.add_argument('--question_path', type=str,
                        required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--output_path', type=str,
                        required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案
    # read and load finance dataset embedding vector
    with open("embedded_dataset/finance6.pickle", 'rb') as f:
        corpus_dict_finance = joblib.load(f)
    # read and load insurance dataset embedding vector
    with open("embedded_dataset/insurance.pickle", 'rb') as f:
        corpus_dict_insurance = joblib.load(f)
    # read and load insurance dataset embedding vector
    with open("embedded_dataset/faq.pickle", 'rb') as f:
        corpus_dict_faq = joblib.load(f)
    # use SentenceTransformer for loading embedding model BAAI/bge-m3
    model = SentenceTransformer("BAAI/bge-m3")
    # use LayerWiseFlagLLMReranker for loading reranker model BAAI/bge-reranker-v2-minicpm-layerwise and run in GPU，can download it by execute the code
    reranker = LayerWiseFlagLLMReranker(
        'BAAI/bge-reranker-v2-minicpm-layerwise', use_fp16=True, device='cuda:0')
    #三種資料這邊code基本上一樣的不用function傳遞以免model重新讀取費時
    for q_dict in tqdm(qs_ref['questions']):
        qs = q_dict['query']
        source = q_dict['source']
        if q_dict['category'] == 'finance':
            """Process finance dataset"""
            filter_corpus = [corpus_dict_finance[int(
                file)][0] for file in q_dict['source']]
            data = [corpus_dict_finance[int(file)][1]
                    for file in q_dict['source']]
            embedded_corpus = []
            datat = []
            idp = []
            k = 0
            #紀錄每一段資料的source id
            for x in filter_corpus:
                embedded_corpus.extend(x)
                tmp = [k]*len(x)
                idp.extend(tmp)
                k += 1
            for x in data:
                datat.extend(x)
            # convert query to embedding vector and run in GPU
            embedded_query = model.encode(qs, convert_to_tensor=True)
            embedded_query = embedded_query.cpu().numpy()
            # calculate text similarity between query and dataset
            similarity = np.dot(embedded_corpus, embedded_query)
            # 由大排到小
            sorted_indices = np.argsort(-similarity)

            relateddoc = [""]*len(source)
            count = 0
            # find the highest similarity score segment from each corpus
            for i in range(len(sorted_indices)):
                docid = idp[sorted_indices[i]]
                if relateddoc[docid] == "":
                    relateddoc[docid] = datat[sorted_indices[i]]
                    count += 1
                if count == len(source):
                    break
            # use reranker to rerank the score
            score = reranker.compute_score(
                [[qs, k] for k in relateddoc], cutoff_layers=[28])

            scoremax = np.argmax(score)

            retrieved = source[scoremax]

            # 將結果加入字典
            answer_dict['answers'].append(
                {"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            """
            Prcoess insurance query
            Same process with fiance
            """
            filter_corpus = [corpus_dict_insurance[int(
                file)][0] for file in q_dict['source']]
            data = [corpus_dict_insurance[int(file)][1]
                    for file in q_dict['source']]
            embedded_corpus = []
            datat = []
            idp = []
            k = 0
            #紀錄每一段資料的source id
            for x in filter_corpus:
                embedded_corpus.extend(x)
                tmp = [k]*len(x)
                idp.extend(tmp)
                k += 1
            for x in data:
                datat.extend(x)
            # convert query to embedding vector and run in GPU
            embedded_query = model.encode(qs, convert_to_tensor=True)
            embedded_query = embedded_query.cpu().numpy()
            similarity = np.dot(embedded_corpus, embedded_query)
            # print(similarity)
            sorted_indices = np.argsort(-similarity)  # 由大排到小

            relateddoc = [""]*len(source)
            count = 0
            # find the highest similarity score segment from each corpus
            for i in range(len(sorted_indices)):
                docid = idp[sorted_indices[i]]
                if relateddoc[docid] == "":
                    relateddoc[docid] = datat[sorted_indices[i]]
                    count += 1
                if count == len(source):
                    break
            score = reranker.compute_score(
                [[qs, k] for k in relateddoc], cutoff_layers=[28])

            scoremax = np.argmax(score)

            retrieved = source[scoremax]

            answer_dict['answers'].append(
                {"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            """
            Process faq query
            Same process with finance
            """
            # 進行檢索
            filter_corpus = [
                corpus_dict_faq[int(file)][0] for file in q_dict['source']]
            data = [corpus_dict_faq[int(file)][1] for file in q_dict['source']]
            embedded_corpus = []
            datat = []
            idp = []
            k = 0
            #紀錄每一段資料的source id
            for x in filter_corpus:
                embedded_corpus.extend(x)
                tmp = [k]*len(x)
                idp.extend(tmp)
                k += 1
            for x in data:
                datat.extend(x)
            # convert query to embedding vector and run in GPU
            embedded_query = model.encode(qs, convert_to_tensor=True)
            embedded_query = embedded_query.cpu().numpy()
            similarity = np.dot(embedded_corpus, embedded_query)
            sorted_indices = np.argsort(-similarity)  # 由大排到小

            relateddoc = [""]*len(source)
            count = 0
            # find the highest similarity score segment from each corpus
            for i in range(len(sorted_indices)):
                docid = idp[sorted_indices[i]]
                if relateddoc[docid] == "":
                    relateddoc[docid] = datat[sorted_indices[i]]
                    count += 1
                if count == len(source):
                    break
            score = reranker.compute_score(
                [[qs, k] for k in relateddoc], cutoff_layers=[28])

            scoremax = np.argmax(score)
            retrieved = source[scoremax]
            answer_dict['answers'].append(
                {"qid": q_dict['qid'], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False,
                  indent=4)  # 儲存檔案，確保格式和非ASCII字符
    end = time.perf_counter()
    print("total time:", end-start)


if __name__ == "__main__":
    '''
    用於資料retrieve，分兩種情況，因reranker運行時間較長，所以沒有reranker的為競賽當天備案，要用哪一個function 將另一個註解掉就可以
    '''
    #執行指令 python flag_retrieve.py --question_path dataset/preliminary/questions_preliminary.json --output_path dataset/preliminary/pred_retrieve.json
    # flag_retrieve_without_reranker()
    flag_retrieve_with_reranker()
