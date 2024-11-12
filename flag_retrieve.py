import os
import json
import argparse
import numpy as np
from tqdm import tqdm 
def flag_retrieve_without_reranker():
    import time
    import joblib
    from sentence_transformers import SentenceTransformer
    start=time.perf_counter()
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    #parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    with open("embedded_dataset/finance6.pickle", 'rb') as f:
        corpus_dict_finance=joblib.load(f)
    with open("embedded_dataset/insurance.pickle", 'rb') as f:
        corpus_dict_insurance=joblib.load(f)
    with open("embedded_dataset/faq.pickle", 'rb') as f:
        corpus_dict_faq=joblib.load(f)
    model = SentenceTransformer("BAAI/bge-m3")
    for q_dict in tqdm(qs_ref['questions']):
        qs=q_dict['query'];source=q_dict['source']
        if q_dict['category'] == 'finance':
            # 進行檢索
            filter_corpus = [corpus_dict_finance[int(file)][0] for file in q_dict['source']]
            embedded_corpus=[]
            idp=[]
            k=0;
            for x in filter_corpus:
                embedded_corpus.extend(x)
                tmp=[k]*len(x)
                idp.extend(tmp)
                k+=1
            embedded_query = model.encode(qs,convert_to_tensor=True)
            embedded_query=embedded_query.cpu().numpy()
            similarity = [np.dot(embedded_query,corpus) for corpus in embedded_corpus ]
            #print(similarity)
            scoremax = np.argmax(similarity) #由大排到小
            #print(qs,source[idp[scoremax]])
            retrieved = source[idp[scoremax]]
            #retrieved = TFIDF_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            # 進行檢索
            filter_corpus = [corpus_dict_insurance[int(file)][0]  for file in q_dict['source']]
            embedded_corpus=[]
            idp=[]
            k=0;
            for x in filter_corpus:
                embedded_corpus.extend(x)
                tmp=[k]*len(x)
                idp.extend(tmp)
                k+=1
            embedded_query = model.encode(qs,convert_to_tensor=True)
            embedded_query=embedded_query.cpu().numpy()
            similarity = [np.dot(embedded_query,corpus) for corpus in embedded_corpus ]
            #print(similarity)
            scoremax = np.argmax(similarity) #由大排到小
            #print(qs,source[idp[scoremax]])
            retrieved = source[idp[scoremax]]
            #retrieved = TFIDF_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            # 進行檢索
            filter_corpus = [corpus_dict_faq[int(file)][0]  for file in q_dict['source']]
            embedded_corpus=[]
            idp=[]
            k=0;
            for x in filter_corpus:
                embedded_corpus.extend(x)
                tmp=[k]*len(x)
                idp.extend(tmp)
                k+=1
            embedded_query = model.encode(qs,convert_to_tensor=True)
            embedded_query=embedded_query.cpu().numpy()
            similarity = [np.dot(embedded_query,corpus) for corpus in embedded_corpus ]
            #print(similarity)
            scoremax = np.argmax(similarity) #由大排到小
            #print(qs,source[idp[scoremax]])
            retrieved = source[idp[scoremax]]
            #retrieved = TFIDF_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
    end=time.perf_counter()
    print("total time:",end-start)
def flag_retrieve_with_reranker():
    import time
    import joblib
    from sentence_transformers import SentenceTransformer
    from FlagEmbedding import LayerWiseFlagLLMReranker
    start=time.perf_counter()
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    #parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    with open("embedded_dataset/finance6.pickle", 'rb') as f:
        corpus_dict_finance=joblib.load(f)
    with open("embedded_dataset/insurance.pickle", 'rb') as f:
        corpus_dict_insurance=joblib.load(f)
    with open("embedded_dataset/faq.pickle", 'rb') as f:
        corpus_dict_faq=joblib.load(f)
    model = SentenceTransformer("BAAI/bge-m3")
    reranker = LayerWiseFlagLLMReranker('BAAI/bge-reranker-v2-minicpm-layerwise', use_fp16=True,device='cuda:0')
    for q_dict in tqdm(qs_ref['questions']):
        qs=q_dict['query'];source=q_dict['source']
        if q_dict['category'] == 'finance':
            # 進行檢索
            filter_corpus = [corpus_dict_finance[int(file)][0] for file in q_dict['source']]
            data = [corpus_dict_finance[int(file)][1] for file in q_dict['source']]
            embedded_corpus=[]
            datat=[]
            idp=[]
            k=0;
            for x in filter_corpus:
                embedded_corpus.extend(x)
                tmp=[k]*len(x)
                idp.extend(tmp)
                k+=1
            for x in data:
                datat.extend(x);
            embedded_query = model.encode(qs,convert_to_tensor=True)
            embedded_query=embedded_query.cpu().numpy()
            similarity = np.dot(embedded_corpus,embedded_query)
            #print(similarity)
            sorted_indices = np.argsort(-similarity) #由大排到小
            
            #topk=np.array(sorted_indices[0:3])
            relateddoc=[""]*len(source)
            count=0
            for i in range(len(sorted_indices)):
                docid=idp[sorted_indices[i]]
                if relateddoc[docid]=="":
                    relateddoc[docid]=datat[sorted_indices[i]]
                    count+=1;
                if count==len(source): break;
            score = reranker.compute_score([[qs, k] for k in relateddoc], cutoff_layers=[28])
            #print(score)

            scoremax=np.argmax(score)
            #ans=sorted_indices[scoremax]
            #print(qs,source[idp[ans]])
            retrieved = source[scoremax]
            #retrieved = TFIDF_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            # 進行檢索
            filter_corpus = [corpus_dict_insurance[int(file)][0] for file in q_dict['source']]
            data = [corpus_dict_insurance[int(file)][1] for file in q_dict['source']]
            embedded_corpus=[]
            datat=[]
            idp=[]
            k=0;
            for x in filter_corpus:
                embedded_corpus.extend(x)
                tmp=[k]*len(x)
                idp.extend(tmp)
                k+=1
            for x in data:
                datat.extend(x);
            embedded_query = model.encode(qs,convert_to_tensor=True)
            embedded_query=embedded_query.cpu().numpy()
            similarity = np.dot(embedded_corpus,embedded_query)
            #print(similarity)
            sorted_indices = np.argsort(-similarity) #由大排到小
            
            relateddoc=[""]*len(source)
            count=0
            for i in range(len(sorted_indices)):
                docid=idp[sorted_indices[i]]
                if relateddoc[docid]=="":
                    relateddoc[docid]=datat[sorted_indices[i]]
                    count+=1;
                if count==len(source): break;
            score = reranker.compute_score([[qs, k] for k in relateddoc], cutoff_layers=[28])
            #print(score)

            scoremax=np.argmax(score)
            #ans=sorted_indices[scoremax]
            #print(qs,source[idp[ans]])
            retrieved = source[scoremax]
            #retrieved = TFIDF_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            # 進行檢索
            filter_corpus = [corpus_dict_faq[int(file)][0] for file in q_dict['source']]
            data = [corpus_dict_faq[int(file)][1] for file in q_dict['source']]
            embedded_corpus=[]
            datat=[]
            idp=[]
            k=0;
            for x in filter_corpus:
                embedded_corpus.extend(x)
                tmp=[k]*len(x)
                idp.extend(tmp)
                k+=1
            for x in data:
                datat.extend(x);
            embedded_query = model.encode(qs,convert_to_tensor=True)
            embedded_query=embedded_query.cpu().numpy()
            similarity = np.dot(embedded_corpus,embedded_query)
            #print(similarity)
            sorted_indices = np.argsort(-similarity) #由大排到小
            
            relateddoc=[""]*len(source)
            count=0
            for i in range(len(sorted_indices)):
                docid=idp[sorted_indices[i]]
                if relateddoc[docid]=="":
                    relateddoc[docid]=datat[sorted_indices[i]]
                    count+=1;
                if count==len(source): break;
            score = reranker.compute_score([[qs, k] for k in relateddoc], cutoff_layers=[28])
            #print(score)

            scoremax=np.argmax(score)
            #ans=sorted_indices[scoremax]
            #print(qs,source[idp[ans]])
            retrieved = source[scoremax]
            #retrieved = TFIDF_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
    end=time.perf_counter()
    print("total time:",end-start)
if __name__ == "__main__":
    #python flag_retrieve.py --question_path dataset/preliminary/questions_example.json --output_path dataset/preliminary/pred_retrieve_flag.json
    #python flag_retrieve.py --question_path dataset/preliminary/questions_preliminary.json --output_path dataset/preliminary/pred_retrieve.json
    #flag_retrieve_without_reranker()
    flag_retrieve_with_reranker()
