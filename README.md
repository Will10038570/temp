# Q1：資源受限環境下之 RGB 影像與 LiDAR 點雲整合及推論最佳化

二面題目 Q1。以輕量級物件偵測模型辨識 RGB 影像中的物件，結合 LiDAR 點雲估計物件距離，並在 GPU 上比較推論最佳化前後的速度、記憶體用量與結果差異。

完整題目見 [Question1_RGB_LiDAR_Inference_Optimization_Assignment.md](Question1_RGB_LiDAR_Inference_Optimization_Assignment.md)。

## 任務摘要

1. **物件辨識與距離估計**
   - 使用具公開預訓練權重的輕量級偵測模型，取得物件框、類別與信心分數。
   - 將 LiDAR 點雲投影到影像，取出每個物件框內的點並估計距離。
2. **效能量測與推論最佳化**
   - 量測原始流程的推論時間、整體處理時間與 GPU 記憶體用量。
   - 至少實作一種最佳化（調整精度、輸入解析度或更換推論引擎），比較前後差異並分析原因。

資料要求：至少 30 組 RGB／LiDAR 配對樣本（例如 [CODa](https://amrl.cs.utexas.edu/coda/)），涵蓋至少 2 種模型支援的類別，並記錄樣本編號。

## 距離估計流程

LiDAR 本身提供每個點的距離，但不知道哪些點屬於哪個物件，因此需要以下步驟：

1. **投影**：用外參（LiDAR→Camera）與內參 K 把 3D 點投影到影像平面。
2. **篩選**：只保留落在偵測框內的點。
3. **去背景**：框內混有後方牆面、地面等點，需濾除（取最近一群、DBSCAN 分群、去除地面點或只取框中心區域）。
4. **彙整**：以中位數、最近群平均或百分位數得出單一距離。
5. **例外處理**：點數不足（物件太遠或太小）時標為無效或改用其他方法。

實際實作的流程、參數與輸出格式見 [src/docs/distance_pipeline.md](src/docs/distance_pipeline.md)。

## 檔案說明

| 檔案 | 說明 |
|---|---|
| `Question1_RGB_LiDAR_Inference_Optimization_Assignment.md` | 題目原文與交付重點 |
| `env_report.md` | 本機 GPU／推論引擎環境檢查報告（含 ORT TensorRT EP 問題與解法） |
| `cnn_netron.py` | 簡單 CNN 範例：匯出 ONNX（`simple_cnn.onnx`）並用 Netron 檢視模型結構 |
| `simple_cnn.onnx` | `cnn_netron.py` 匯出的模型，固定輸入 shape `(8, 3, 224, 224)` |
| `test.py` | 用 torchviz 畫出簡單計算圖（輸出 `computation_graph.svg`） |
| `新增 文字文件.txt` | 距離估計方法的筆記（內容已整理於上方「距離估計流程」） |

## 執行環境

詳見 [env_report.md](env_report.md)。重點：

- GPU：NVIDIA GeForce RTX 5070 Laptop GPU（Blackwell，sm_120），8 GB VRAM
- venv：`李柏禹_二面題目\.venv`（位於上一層資料夾）
- 主要套件：torch 2.14.1+cu130、onnxruntime-gpu 1.31.0、TensorRT 10.16.1.11（cu13）
- 可比較的推論引擎：PyTorch、ORT + CUDA EP、ORT + TensorRT EP、TensorRT 原生 API

### 使用 ORT TensorRT EP 的注意事項

ORT 1.31 需要 TensorRT 10，且找不到 `site-packages\tensorrt_libs\` 中的 DLL。建立 session 前先 `import tensorrt`（或把 `tensorrt_libs` 加進 `PATH`），否則會**默默退回 CPU**。每個 session 建立後都應檢查：

```python
import tensorrt  # 必須在建立 ORT session 前載入
import onnxruntime as ort

sess = ort.InferenceSession("model.onnx", providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"])
assert sess.get_providers()[0] == "TensorrtExecutionProvider", sess.get_providers()
```

TensorRT `.engine` 檔綁定 TensorRT 版本與 GPU 架構，換環境（例如 Colab T4）需重建。

## 範例腳本

```powershell
..\.venv\Scripts\Activate.ps1
python cnn_netron.py   # 匯出 simple_cnn.onnx 並在瀏覽器開啟 Netron（Ctrl+C 結束）
python test.py         # 產生 computation_graph.svg（需安裝 Graphviz）
```

## 交付重點

1. 模型與資料選擇（架構、權重來源、前處理、為何適合邊緣裝置）
2. RGB／LiDAR 整合（投影、篩選、背景點處理，附點雲疊圖）
3. 模型推論與最佳化（引擎、精度、解析度，確認在 GPU 執行）
4. 改善前後比較（推論時間、GPU 記憶體、各階段耗時與主要瓶頸）
5. 結果驗證（至少檢查 10 個物件的類別、框與距離）
6. 最終設定建議
