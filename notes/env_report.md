# 環境檢查報告：ONNX Runtime TensorRT EP 無法使用

檢查日期：2026-10-24

## 1. 硬體與驅動

| 項目 | 內容 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU（Blackwell，sm_120） |
| VRAM | 8 GB |
| 驅動版本 | 581.91（支援 CUDA 13.0） |
| CUDA Toolkit | 未安裝（不需要，pip 套件已附帶 runtime DLL） |
| 作業系統 | Windows 11 Home 10.0.26100 |

## 2. Python 環境

venv 路徑：`李柏禹_二面題目\.venv`

| 套件 | 版本 |
|---|---|
| torch | 2.14.1+cu130 |
| torchvision | 0.29.1+cu130 |
| onnxruntime-gpu | 1.31.0 |
| onnx | 已安裝（匯出 ONNX 需要） |
| tensorrt-cu13 / -libs / -bindings | 10.16.1.11（原為 11.4.0.106，已依 5.A 降版） |
| nvidia-cuda-runtime | 13.4.92 |
| nvidia-cudnn-cu13 | 9.27.0.42 |
| nvidia-cublas | 13.8.0.4 |

## 3. 實際推論測試

用一個小型卷積網路（Conv-ReLU-Conv，輸入 1×3×64×64）在各引擎上執行，並和 CPU 結果比較。

| 引擎 | 結果 | 實際執行裝置 | 與 CPU 結果的最大誤差 |
|---|---|---|---|
| PyTorch | 正常 | GPU（cuDNN 9.24，支援 sm_120） | 3.9e-4 |
| ONNX Runtime + CUDA EP | 正常 | GPU | 3.9e-4 |
| **ONNX Runtime + TensorRT EP** | **失敗** | **自動退回 CPU** | — |
| TensorRT 原生 API | 正常 | GPU | 5.5e-4 |

1e-4 等級的誤差是 GPU 和 CPU 浮點運算順序不同造成的，屬於正常範圍。

## 4. 問題：ORT TensorRT EP 失敗並默默退回 CPU

### 錯誤訊息

```
EP Error ...onnxruntime_pybind_state.cc:729 onnxruntime::python::RegisterTensorRTPluginsAsCustomOps
Please install TensorRT libraries as mentioned in the GPU requirements page, make sure they're
in the PATH or LD_LIBRARY_PATH, and that your GPU is supported.
```

建立 session 時指定了 `TensorrtExecutionProvider`，但 `session.get_providers()[0]` 回傳的是 `CPUExecutionProvider`。

### 原因：TensorRT 主版本不相容

| | 需要 / 提供的 DLL |
|---|---|
| ORT 1.31 的 `onnxruntime_providers_tensorrt.dll` 需要 | `nvinfer_10.dll`、`nvonnxparser_10.dll`（TensorRT **10**） |
| 已安裝的 `tensorrt_cu13_libs` 11.4 提供 | `nvinfer_11.dll`、`nvinfer_plugin_11.dll` …（TensorRT **11**） |

DLL 名稱裡帶有主版本號，ORT 找不到 `nvinfer_10.dll`，TensorRT EP 就無法載入。

`ort.get_available_providers()` 仍會列出 `TensorrtExecutionProvider`，因為它只代表這個 EP 有編譯進 ORT，不代表執行時載入得到 TensorRT 函式庫。

ONNX Runtime 官網 TensorRT EP 頁面的相容性表只列到 ORT 1.22 / TensorRT 10.9，沒有寫 1.31 對應哪一版 TensorRT。上面的結論來自實際檢查 DLL 的相依關係。

### 對作業的影響

fallback 發生時程式**不會中斷**，只會印出一段錯誤訊息。如果沒注意到，會以為在跑 TensorRT，其實在跑 CPU，量到的推論時間就完全錯了。

**對策**：建立每個 session 後都檢查一次

```python
assert sess.get_providers()[0] == "TensorrtExecutionProvider", sess.get_providers()
```

## 5. 解決方式

### A. 把 TensorRT 降到 10.x（建議）

原生 TensorRT 和 ORT TensorRT EP 都能用，可比較的組合最多：PyTorch / ORT-CUDA / ORT-TRT / 原生 TRT。

```powershell
pip install "tensorrt-cu13==10.16.1.11" "tensorrt-cu13-libs==10.16.1.11" "tensorrt-cu13-bindings==10.16.1.11"
```

PyPI 上 `tensorrt-cu13-libs` 的 10.x 版本有：10.16.1.11、10.16.0.72、10.15.1.29、10.14.1.48、10.13.3.9、10.13.2.6。

降版後要重跑測試，確認 ORT TensorRT EP 確實在 GPU 上執行。

**已執行（2026-10-24），結果：**

只降版還不夠。`nvinfer_10.dll` 等檔案裝在 `site-packages\tensorrt_libs\`，ORT 不會去這個資料夾找，所以仍然退回 CPU。建立 session 前做以下任一種處理就能解決，兩種都實測有效：

```python
import tensorrt          # 做法 1：在建立 ORT session 前先 import，DLL 會先載入 process
import onnxruntime as ort
```

```python
import os, tensorrt_libs # 做法 2：把 tensorrt_libs 加進 PATH
os.environ["PATH"] = os.path.dirname(tensorrt_libs.__file__) + os.pathsep + os.environ["PATH"]
import onnxruntime as ort
```

| 引擎（TensorRT 10.16） | 實際執行裝置 | 與 CPU 結果的最大誤差 |
|---|---|---|
| PyTorch | GPU | 4.0e-4 |
| ORT + CUDA EP | GPU | 4.0e-4 |
| ORT + TensorRT EP | GPU（`TensorrtExecutionProvider`） | 4.5e-4 ~ 4.7e-4 |
| TensorRT 原生 API | GPU | 4.6e-4 |

### B. 維持 TensorRT 11

不用 ORT TensorRT EP，改用「原生 TensorRT API」和「ORT CUDA EP」做比較。作業只要求至少一種最佳化方法，這樣也足夠。

## 6. 其他備註

- **CUDA minor version compatibility**：CUDA runtime 13.4 能在只支援到 13.0 的驅動上正常執行，因為主版本相同（13），minor 版本較新也相容。要遵守的規則是「驅動支援的主版本 ≥ runtime 的主版本」。
- **TensorRT engine 不能跨環境使用**：`.engine` 檔綁定 TensorRT 版本和 GPU 架構。換 TensorRT 版本或換到 Colab（例如 T4，sm_75）都要重建。
- **測試腳本**：`C:\Users\swill\AppData\Local\Temp\claude\C--Users-swill-Desktop-fox----------Q1\b4f94a2d-e37f-4727-bed2-9bc312ba013b\scratchpad\gpu_check.py`。這是暫存位置，要保留的話請複製到專案資料夾。
