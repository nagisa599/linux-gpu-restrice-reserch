import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# データサイズを増やして、より多くのGPUリソースを使用
data1 = torch.randn((385, 385), device=device)
data2 = torch.randn((385, 385), device=device)

while True:
    with torch.cuda.stream(stream1):
        # より複雑な演算を行う
        result1 = torch.matmul(data1, data1)

    with torch.cuda.stream(stream2):
        # より複雑な演算を行う
        result2 = torch.matmul(data2, data2)

    torch.cuda.synchronize()  # 同期を保持
