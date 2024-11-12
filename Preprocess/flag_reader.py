import os
import json
import argparse
import numpy as np
from tqdm import tqdm
# 用於中文文本分詞
import jieba

def remove_number(text):
    '''
        僅用在處理Finance資料，因為finance裡有比較沒有用處的表格數據資料(單純數字對retrieve比較沒用)，所以將數字、刪除
        回傳清理乾淨的資料
    '''
    #中文分詞，將數字分開
    words = jieba.lcut(text)
    #去空格、換行
    words = [w for w in words if w.strip()]
    
    result = []
    #將除了年月日以外的純數字清除:
    for i in range(len(words)):

        current_word = words[i]

        if current_word.isdigit():

            if i + 1 < len(words) and any(w in words[i + 1] for w in ["年", "月", "日"]):
                result.append(current_word)

        else:
            result.append(current_word)
    result = "".join(result)
    #回傳清理乾淨的資料
    return result
def load_data(source_path):
    '''
        讀取資料，用multiprocessing 加速pdf讀取，返回dict，key為檔案名稱，value為PDF檔內容的文本
    '''
    import multiprocessing
    #拿取cpu數
    num_processes = multiprocessing.cpu_count() 
    #建立multiprocessing pool
    pool = multiprocessing.Pool(num_processes)
    # 獲取資料夾中的檔案列表
    masked_file_ls = os.listdir(source_path)  
    #因為multiprocessing，所以要把key值也傳入，否則會亂掉
    texts= pool.starmap(read_pdf, {(os.path.join(source_path, file),int(file.replace('.pdf', ''))) for file in tqdm(masked_file_ls)})
    #結束multiprocessing
    pool.close()
    pool.join()
    #建立dict
    corpus_dict={texts[i][0]: texts[i][1] for i in range(len(texts))}
    #回傳dict
    return corpus_dict



def read_pdf(pdf_loc, key, page_infos: list = None):
    '''
    讀取每個pdf資料，部分使用OCR讀取，回傳聞本內容
    '''
    import io
    import fitz
    import pytesseract
    from PIL import Image
    #打開指定的PDF文件
    pdf = fitz.open(pdf_loc) 
    #找到tesseract檔案位置，可從https://digi.bib.uni-mannheim.de/tesseract/ 下載
    #用於OCR
    pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    #初始化string
    pdf_text = ''
    for idx, page in enumerate(pdf):  # 迴圈遍歷每一頁
        # 提取頁面的文本內容
        text = page.get_text()
        #將文字append到回傳值
        if text:
            pdf_text += text
        else:
            #如果沒有讀到文字，代表為掃描式pdf，改用ocr
            #拿取圖片
            images=page.get_images()
            image_text=''
            for image in images:
                #對每個image 做OCR拿資料
                imgposition=image[0]
                image_base=pdf.extract_image(imgposition)
                image_path = image_base["image"]
                image_PIL = Image.open(io.BytesIO(image_path))
                #只有在跑finance資料會有以下這行，處理不同方向pdf，人工發現的
                if key==652 or key==753 or key==979: image_PIL = image_PIL.rotate(-90, expand=True) 
                #抓取文字by OCR
                image_text += pytesseract.image_to_string(image_PIL,lang='chi_tra')
            #appen text
            if image_text:
                pdf_text += image_text
        
    pdf.close()  # 關閉PDF文件

    #以下兩行只有在處理finance資料會有，用於處理符號、數字
    pdf_text=remove_number(pdf_text)
    pdf_text=pdf_text.replace("-","").replace("$","").replace(",","")

    return [key,pdf_text]  # 返回萃取出的文本


if __name__ == "__main__":
    '''
    主要進行資料前處理
    '''
    #執行程式碼指令 python flag_reader.py  --source_path reference 
    import time
    import joblib
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer
    #用於確認程式執行時間
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
    #處理OCR難以判定的資料
    data_dict = {}
    with open("finance.txt", 'r', encoding='utf-8') as file:
        for line in file:
            # Split each line at the first colon
            key, value = line.split(':', 1)
            # Strip any whitespace and store in the dictionary
            data_dict[int(key.strip())] = value.strip()
    #將人工資料append到dict
    for key,data in data_dict.items():
        corpus_dict_finance[key]=data_dict[key]+corpus_dict_finance[key]
        print(key,corpus_dict_finance[key])
    #用此embedding model，run code 會自己下載model
    model = SentenceTransformer("BAAI/bge-m3")
    #處理finance資料
    for key,data in tqdm(corpus_dict_finance.items()):
        #資料分段用兩種不同的來分，以觀察發現能獲得不同的資訊(因為問題關鍵字相距可能不一)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=300)
        text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=150)
        prefit_corpus=[]
        #把每個文章切割，並設定id(從哪篇文章來)，方便後續使用
        tmp=text_splitter.split_text(data)
        tmp2=text_splitter2.split_text(data)
        prefit_corpus.extend(tmp)
        prefit_corpus.extend(tmp2)
        #embedding data
        embeddings=model.encode(prefit_corpus,convert_to_tensor=True)
        #轉成np array
        embeddings=embeddings.cpu().numpy()
        #存到dict
        corpus_dict_finance[key]=(embeddings,prefit_corpus)
    #存到pickle檔，這樣競賽當天能直接使用
    with open("embedded_dataset/finance6.pickle", 'wb') as f:
        joblib.dump(corpus_dict_finance,f) 
    #用於確認完成進度
    print("finance completed!!!")
    #處理insurance資料
    for key,data in tqdm(corpus_dict_insurance.items()):
        #資料分段用兩種不同的來分，以觀察發現能獲得不同的資訊(與finance形式不同所以切法不同)(因為問題關鍵字相距可能不一)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
        text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=400)
        prefit_corpus=[]
        #把每個文章切割，並設定id(從哪篇文章來)，方便後續使用
        tmp=text_splitter.split_text(data)
        tmp2=text_splitter2.split_text(data)
        prefit_corpus.extend(tmp)
        prefit_corpus.extend(tmp2)
        #embedding data
        embeddings=model.encode(prefit_corpus,convert_to_tensor=True)
        #轉成np array
        embeddings=embeddings.cpu().numpy()
        #存到dict
        corpus_dict_insurance[key]=(embeddings,prefit_corpus)
    #存到pickle檔，這樣競賽當天能直接使用
    with open("embedded_dataset/insurance.pickle", 'wb') as f:
        joblib.dump(corpus_dict_insurance,f)  
    #用於確認完成進度
    print("insurance completed!!!")
    #處理faq資料
    for key,data in tqdm(key_to_source_dict.items()):
        #直接使用data，不用分段，因為faq本身就比較短
        prefit_corpus=data
        #embedding
        embeddings=model.encode(prefit_corpus,convert_to_tensor=True)
        embeddings=embeddings.cpu().numpy()
        key_to_source_dict[key]=(embeddings,prefit_corpus)
    #存到pickle檔，這樣競賽當天能直接使用
    with open("embedded_dataset/faq.pickle", 'wb') as f:
        joblib.dump(key_to_source_dict,f)  
    #用於確認完成進度
    print("faq completed!!!")
    #用於確認執行時間
    end=time.perf_counter()
    print("total time:",end-start)
