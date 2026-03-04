1. 启动网页服务：
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

2. 启动 HTTP 穿透(在另一个终端)
    ```bash
    cpolar http 8000
    ```

3. 然后基于可以用cpolar输出的网址访问了！上传的音频文件会存放在 `uploads/` 目录下。