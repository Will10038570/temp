import torch
from torch import nn
import netron

# 簡單 CNN：3 個卷積區塊 + 全域池化 + 分類頭
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # (N, 3, 224, 224) → (N, 16, 112, 112)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # → (N, 32, 56, 56)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # → (N, 64, 28, 28)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # → (N, 64, 1, 1)
            nn.Flatten(),             # → (N, 64)
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = SimpleCNN().eval()

# shape：[8, 3, 224, 224]
images = torch.randn(8, 3, 224, 224)

with torch.inference_mode():
    print("Output shape:", tuple(model(images).shape))

# 匯出 ONNX，Netron 可直接開啟
onnx_path = "simple_cnn.onnx"
torch.onnx.export(
    model,
    (images,),
    onnx_path,
    input_names=["images"],
    output_names=["scores"],
    # 不設 dynamic_axes：shape 固定為 images 的 (8, 3, 224, 224)
    dynamo=False,
)
print(f"Saved {onnx_path}")

# 啟動 Netron，會自動在瀏覽器開啟；按 Ctrl+C 結束
netron.start(onnx_path)
netron.wait()
