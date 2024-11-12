import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import jieba
  # 用於中文文本分詞
#import pdfplumber  # 用於從PDF文件中提取文字的工具
#from rank_bm25 import BM25Okapi  # 使用BM25演算法進行文件檢索
def remove_number(text):
    words = jieba.lcut(text)
    words = [w for w in words if w.strip()]
    # print(words)
    result = []

    for i in range(len(words)):

        current_word = words[i]

        if current_word.isdigit():

            if i + 1 < len(words) and any(w in words[i + 1] for w in ["年", "月", "日"]):
                result.append(current_word)

        else:
            result.append(current_word)
    result = "".join(result)

    return result
# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path):
    import multiprocessing
    num_processes = multiprocessing.cpu_count() 
    pool = multiprocessing.Pool(num_processes)
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    texts= pool.starmap(read_pdf, {(os.path.join(source_path, file),int(file.replace('.pdf', ''))) for file in tqdm(masked_file_ls)})
    pool.close()
    pool.join()
    corpus_dict={texts[i][0]: texts[i][1] for i in range(len(texts))}
    
    return corpus_dict


# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, key, page_infos: list = None):
    import io
    import fitz
    import pytesseract
    from PIL import Image
    # Open some document, for example a PDF (could also be EPUB, XPS, etc.)
 
    #reads table from pdf file
    pdf = fitz.open(pdf_loc)  # 打開指定的PDF文件
    #print(key)
    pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理
    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    #pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for idx, page in enumerate(pdf):  # 迴圈遍歷每一頁
        #text = page.extract_text()  # 提取頁面的文本內容
        text = page.get_text()
        if text:
            pdf_text += text
        else:
            images=page.get_images()
            image_text=''
            for image in images:
                imgposition=image[0]
                image_base=pdf.extract_image(imgposition)
                image_path = image_base["image"]
                image_PIL = Image.open(io.BytesIO(image_path))
                if key==652 or key==753 or key==979: image_PIL = image_PIL.rotate(-90, expand=True) #just for finance
                image_text += pytesseract.image_to_string(image_PIL,lang='chi_tra')
            if image_text:
                pdf_text += image_text
        
    pdf.close()  # 關閉PDF文件

    #just for finance
    pdf_text=remove_number(pdf_text)
    pdf_text=pdf_text.replace("-","").replace("$","").replace(",","")

    return [key,pdf_text]  # 返回萃取出的文本


if __name__ == "__main__":
    #python flag_reader.py  --source_path reference 
    import time
    import joblib
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer
    #import torch
    #from torch import torch
    #import gc
    start=time.perf_counter()
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑

    args = parser.parse_args()  # 解析參數

    source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
    corpus_dict_insurance = load_data(source_path_insurance)

    source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
    corpus_dict_finance = load_data(source_path_finance)

    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
    data_dict = {}
    with open("finance.txt", 'r', encoding='utf-8') as file:
        for line in file:
            # Split each line at the first colon
            key, value = line.split(':', 1)
            # Strip any whitespace and store in the dictionary
            data_dict[int(key.strip())] = value.strip()
    for key,data in data_dict.items():
        corpus_dict_finance[key]=data_dict[key]+corpus_dict_finance[key]
        print(key,corpus_dict_finance[key])
    #print(corpus_dict_finance[699])
    #print("test1")
    model = SentenceTransformer("BAAI/bge-m3")
    #print("test2")
    for key,data in tqdm(corpus_dict_finance.items()):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=300)
        text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=150)
        prefit_corpus=[]
        #把每個文章切割，並設定id(從哪篇文章來)，方便後續使用
        tmp=text_splitter.split_text(data)
        tmp2=text_splitter2.split_text(data)
        prefit_corpus.extend(tmp)
        prefit_corpus.extend(tmp2)
        embeddings=model.encode(prefit_corpus,convert_to_tensor=True)
        embeddings=embeddings.cpu().numpy()
        corpus_dict_finance[key]=(embeddings,prefit_corpus)
    
    with open("embedded_dataset/finance6.pickle", 'wb') as f:
        joblib.dump(corpus_dict_finance,f) 
    print("finance completed!!!")

    for key,data in tqdm(corpus_dict_insurance.items()):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
        text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=400)
        prefit_corpus=[]
        #把每個文章切割，並設定id(從哪篇文章來)，方便後續使用
        tmp=text_splitter.split_text(data)
        tmp2=text_splitter2.split_text(data)
        prefit_corpus.extend(tmp)
        prefit_corpus.extend(tmp2)
        embeddings=model.encode(prefit_corpus,convert_to_tensor=True)
        embeddings=embeddings.cpu().numpy()
        corpus_dict_insurance[key]=(embeddings,prefit_corpus)
    with open("embedded_dataset/insurance2.pickle", 'wb') as f:
        joblib.dump(corpus_dict_insurance,f)  
    print("insurance completed!!!")
    for key,data in tqdm(key_to_source_dict.items()):
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
        prefit_corpus=data
        #把每個文章切割，並設定id(從哪篇文章來)，方便後續使用
        embeddings=model.encode(prefit_corpus,convert_to_tensor=True)
        embeddings=embeddings.cpu().numpy()
        key_to_source_dict[key]=(embeddings,prefit_corpus)
    
    with open("embedded_dataset/faq.pickle", 'wb') as f:
        joblib.dump(key_to_source_dict,f)  
    print("faq completed!!!")

    end=time.perf_counter()
    print("total time:",end-start)
