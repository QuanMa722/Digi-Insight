"""
准确率 (Accuracy)：分类正确的样本数占总样本数的比例。
其中，TP（True Positive）为真正例，TN（True Negative）为真反例，FP（False Positive）为假正例，FN（False Negative）为假反例。
精确率 (Precision)：预测为正类的样本中实际为正类的比例。
召回率 (Recall)：实际为正类的样本中被正确预测为正类的比例。
F1 分数 (F1 Score)：精确率和召回率的调和平均数。
特异度 (Specificity)：实际为负类的样本中被正确预测为负类的比例。
AUC-ROC (Area Under the Receiver Operating Characteristic Curve)：ROC 曲线下的面积，衡量分类器在所有可能的分类阈值下的性能。值越接近 1 表示模型越好。
AUC-PR (Area Under the Precision-Recall Curve)：精确率-召回率曲线下的面积，适用于类别不平衡的情况。
Kappa 统计量 (Cohen's Kappa)：考虑到偶然的准确率，并衡量分类器的一致性。
错误率 (Error Rate)：分类错误的样本占总样本数的比例。
Matthews 相关系数 (Matthews Correlation Coefficient, MCC)：综合考虑 TP、TN、FP 和 FN 的度量，适用于不平衡的分类问题。
"""