# Preprocess
## 執行程式
執行程式碼指令 python flag_reader.py  --source_path reference
說明:程式是可以將finance insurance faq部分分開處理的，運行時把其他兩個的code註解掉即可
## tesseract OCR使用
https://digi.bib.uni-mannheim.de/tesseract/ 
## 人工處理資料
在finance.txt，主要用於OCR無法辨識的資料
## embedding model
"BAAI/bge-m3"
執行到此code即可下載model

model = SentenceTransformer("BAAI/bge-m3")

## 注意事項
這幾行code只有在處理finance時執行，程式註解有說明

pdf_text=remove_number(pdf_text)
pdf_text=pdf_text.replace("-","").replace("$","").replace(",","")

if key==652 or key==753 or key==979: image_PIL = image_PIL.rotate(-90, expand=True) 

