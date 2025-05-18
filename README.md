# Word Rotator's Distance with Subspace Optimal Transport

このプロジェクトは、異なるサイズの特徴量を対象としたGPU並列化可能な部分空間最適輸送手法と、Word Rotator's Distance（WRD）を組み合わせた類似度の試みです。

## 背景

本プロジェクトは、以下の2つの研究を基にしています。

### Word Rotator's Distance

この研究では、単語ベクトルのノルムと方向を分離し、最適輸送理論を用いてテキスト間の意味的類似度を測定する新しい手法が提案されています。

### 異なるサイズの特徴量を対象としたGPU並列化可能な部分空間最適輸送手法の検討

この研究では、異なるサイズの確率分布間での最適輸送問題を、GPU並列化可能な部分空間最適輸送（SOT）問題に変換する手法が提案されています。

## 特徴

- Word Rotator's Distanceによる高精度な意味的類似度評価をGPUによって並列的に行うことができます。

## 引用

- 黄健明, 笠井裕之. "異なるサイズの特徴量を対象としたGPU並列化可能な部分空間最適輸送手法の検討." 2023年度人工知能学会全国大会論文集.  
  [https://doi.org/10.11517/pjsai.JSAI2023.0_2T4GS504](https://doi.org/10.11517/pjsai.JSAI2023.0_2T4GS504)

- Yokoi, Sho, et al. "Word Rotator's Distance." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020.  
  [https://doi.org/10.18653/v1/2020.emnlp-main.236](https://doi.org/10.18653/v1/2020.emnlp-main.236)
