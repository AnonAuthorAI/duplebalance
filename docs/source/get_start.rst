Getting Started
***************

Background
====================================

Class-imbalance (also known as the long-tail problem) is the fact that the 
classes are not represented equally in a classification problem, which is 
quite common in practice. For instance, fraud detection, prediction of 
rare adverse drug reactions and prediction gene families. Failure to account 
for the class imbalance often causes inaccurate and decreased predictive 
performance of many classification algorithms. 

Imbalanced learning (IL) aims 
to tackle the class imbalance problem to learn an unbiased model from 
imbalanced data. This is usually achieved by changing the training data 
distribution by resampling or reweighting. However, naive resampling or 
reweighting may introduce bias/variance to the training data, especially 
when the data has class-overlapping or contains noise.

Ensemble imbalanced learning (EIL) is known to effectively improve typical 
IL solutions by combining the outputs of multiple classifiers, thereby 
reducing the variance introduce by resampling/reweighting. 

About `duplebalance`
====================================

Imbalanced Learning (IL) is an important problem that widely exists in data 
mining applications. Typical IL methods utilize intuitive class-wise resampling 
or reweighting to directly balance the training set. However, some recent 
research efforts in specific domains show that class-imbalanced learning can 
be achieved without class-wise manipulation. This prompts us to think about 
the relationship between the two different IL strategies and the nature of 
the class imbalance. Fundamentally, they correspond to two essential imbalances 
that exist in IL: the difference in quantity between examples from different 
classes as well as between easy and hard examples within a single class, 
i.e.,inter-class and intra-class imbalance. Existing works fail to explicitly 
take both imbalances into account and thus suffer from suboptimal performance. 
In light of this, we present Duple-Balanced Ensemble, namely DUBE, a versatile 
ensemble learning framework. Unlike prevailing methods, DUBE directly performs 
inter-class and intra-class balancing without relying on heavy distance-based 
computation, which allows it to achieve competitive performance while being 
computationally efficient. Code, documentation, and examples 
are available at `github.com/AnonAuthorAI/duplebalance <https://github.com/AnonAuthorAI/duplebalance>`__.