# Phân loại văn bản tiếng Việt

Đây là repo phân loại văn bản tiếng Việt

## Tính năng

- Dữ liệu tiếng Việt
- Sử dụng mô hình BERT để
- Sử mạng kiến trúc mạng LSTM

## Cài đặt

Yêu cầu [python](https://www.python.org/) >= 3.6.

Download mô hình để sử dụng và cho vào thư mục model
```sh
https://drive.google.com/file/d/10WFCHP9GTER41xUsaGGxNg14EXAIY2to/view?usp=drive_link
```

Cài đặt các thư viện cần thiết để khởi chạy server.
```sh
git clone https://github.com/hieunguyenquoc/vietnam-text-classification.git
cd text-classification-pytorch
pip install -r requirements.txt
python ./src/download_BERT_embedding_model.py
python ./src/main.py
```

Huấn luyện
```sh
python ./src/training.py
```