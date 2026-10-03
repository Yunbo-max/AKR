# 追加图的冻结数据与原始图稿

本目录保留上一轮七张图的数值快照、CSV、原绘图程序、PNG和SVG。
最终整合论文使用 `../figures/analysis/` 下同名的重新排版版本，数据未变；全部已在 main.tex 中引用。
最终重绘入口是 `../plots/make_integrated_analysis.py`，会生成PDF/PNG/SVG。

定义：M[s,t]为source训练读出在target测试音频上的准确率；D[s,t]=M[t,t]-M[s,t]。
图中保留负值；Full不等于LP8；两个prompt不当作独立数据；class-routed结果不替代continuous AKR。

没有运行新实验。既有压缩结构不被解释为已经测得的held-out梯度可预测性。
最终论文位置、所有新图prompt和首个完整39-query案例见项目根目录。
