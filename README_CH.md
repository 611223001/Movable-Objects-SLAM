# SLAM With Movable Objects In Interactive Environment

![Demo GIF](show/MO_SLAM(object-move).gif)



## 安裝說明

```shell
git clone https://github.com/611223001/Movable-Objects-SLAM.git
```

1.建立conda環境
```shell
conda create -n mo_slam python=3.10 -y
conda activate mo_slam
```

2.安裝 [detectron2](https://github.com/facebookresearch/detectron2)
```shell
pip install torch torchvision torchaudio

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

```



3.安裝其他套件
```shell
cd MovableObjectSLAM

# MovableObjectSLAM$ 
conda env update -n mo_slam -f environment.yml --prune
```

### 測試

```shell
# MovableObjectSLAM$ 
python main.py
```

## 簡介

本專案實作一套用於互動式室內環境的單目 SLAM 系統，目標是在物件可能被移動、並於之後再次恢復靜態的情況下，仍能維持地圖中的物件一致性。

不同於傳統 SLAM 方法常將動態物件視為干擾並直接排除，本系統將可動物件納入地圖表示，透過物件層級建模與狀態管理，維持物件在移動前後的身分一致性。系統可在物件移動期間暫時隔離不穩定觀測，並於物件恢復靜態後重新估計其位姿，使其重新整合進地圖之中。


### 特色

- 支援物件層級地圖表示與身分一致性維護
- 可處理物件由靜態、移動到再次靜態的狀態轉換
- 結合幾何一致性進行物件位姿重新估計
- 建立於特徵點式單目 SLAM 架構之上
- 適用於包含可動物件的互動式室內場景

### 核心概念

本系統並非在物件移動後直接將其視為新物件，而是透過以下流程維持地圖一致性：

1. 偵測物件是否發生移動
2. 暫時排除不可靠的動態觀測
3. 在物件恢復靜態後重新估計其位姿
4. 將物件與其相關地標重新整合回地圖

藉此，系統能在互動式環境中維持可重複利用且具一致性的地圖表示。

### 系統限制

這個方法雖然嘗試把可動物件納入地圖中，但本質上仍建立在單目特徵點 SLAM 上，而這類架構並不適合穩定地處理物件層級表示。

由於特徵點式單目 SLAM 所建立的地圖點是稀疏的，且僅反映局部影像特徵，難以有效描述完整物件結構，因此需要額外引入物件模型進行補充。然而，這樣的設計與原本以地標為核心的地圖表示並不完全相容，也使得系統難以自然整合 loop closing。

此外，系統目前未整合 loop closing，因此在長時間運行下可能產生累積漂移。整體效果也依賴影像分割品質與物件觀測條件，當分割不穩定或觀測不足時，結果會受到影響。

因此，這個方法較偏向一種可行性探索，而非成熟穩定的物件 SLAM 解法。

### 未來方向
未來若要提升穩定性與擴充性，較合理的方向是從地圖表示與感知方式本身進一步調整。

其中一個方向是使用通用視覺模型。相較於依賴固定類別與分割結果的做法，通用視覺模型有機會提供更穩定的物件特徵與跨視角對應能力，讓物件追蹤、身分維持與狀態判斷不必過度依賴目前較脆弱的分割與局部幾何條件。

另一個方向是改用高斯潑濺 SLAM 為基底。相較於特徵點式 SLAM 只建立稀疏地圖點，高斯潑濺方法能提供更接近物件表面與外觀的表示，理論上也更適合直接支援物件層級建模。若未來能將可動物件的一致性機制建立在這類表示上，系統在物件追蹤、重建與地圖一致性方面，應該會比目前的特徵點式+額外物件模型架構更自然。
