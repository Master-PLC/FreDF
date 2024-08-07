# Learning to Forecast in the Transformed Domain (FreDF)
FreDF is an open-source library for deep learning researchers, especially for deep time series analysis.

We provide a neat code base to evaluate advanced deep time series models or develop your model on transformed domain, which covers three mainstream tasks: **long- and short-term forecasting, and imputation.**

🚩**News** (2023.12) We add implementations to train and evaluate deep learning models within transformed domain (Frequency Domain) on three main tasks.

## Leaderboard for Time Series Analysis

Till October 2023, the top three models for five different tasks are:

| Model`<br>`Ranking | Long-term`<br>`Forecasting                   | Short-term`<br>`Forecasting                                                          | Imputation                                                                             |
| -------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | 
| 🥇 1st               | FreDF + [iTrans.](https://arxiv.org/abs/2310.06625)  | FreDF + [FreTS](https://arxiv.org/abs/2311.06184)                                              | FreDF + [iTrans.](https://arxiv.org/abs/2310.06625)                                              |

**Note: We will keep updating this leaderboard.** If you have proposed advanced and awesome models, you can send us your paper/code link or raise a pull request. We will add them to this repo and update the leaderboard as soon as possible.

**Compared models of this leaderboard.** ☑ means that their codes have already been included in this repo.

- ☑ **iTransformer** - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[arXiv 2023]](https://arxiv.org/abs/2310.06625) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
- ☑ **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
- ☑ **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py).
- ☑ **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py).
- ☑ **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]](https://proceedings.mlr.press/v162/zhou22g.html) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py).
- ☑ **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py).
- ☑ **Transformer** - Attention is All You Need [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py).
- ☑ **TiDE** - Long-term Forecasting with TiDE: Time-series Dense Encoder [[arXiv 2023]](https://arxiv.org/pdf/2304.08424.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TiDE.py).
- ☑ **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Crossformer.py).
- ☑ **TCN**
- ☑ **LSTM**

See our latest paper [[FreDF]](https://arxiv.org/abs/2402.02399) for the comprehensive benchmark. We will release a real-time updated online version soon.

## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder `./dataset`. Here is a summary of supported datasets.

<p align="center">
<img src=".\pic\dataset-fredf.jpg" height = "200" alt="" align=center />
</p>

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# long-term forecast
bash ./scripts/fredf_exp/ltf_overall/ETTh1_script/iTransformer.sh
# short-term forecast
bash ./scripts/fredf_exp/stf_overall/FreTS_M4.sh
# imputation
bash ./scripts/fredf_exp/imp_autoencoder/ETTh1_script/iTransformer.sh
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/iTransformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.


## Acknowledgement

This library is mainly constructed based on the following repos, following the training-evaluation pipelines and the implementation of baseline models:

- Time-Series-Library: https://github.com/thuml/Time-Series-Library.

All the experiment datasets are public, and we obtain them from the following links:

- Long-term Forecasting and Imputation: https://github.com/thuml/Autoformer.
- Short-term Forecasting: https://github.com/ServiceNow/N-BEATS.

## All Thanks To Our Contributors
