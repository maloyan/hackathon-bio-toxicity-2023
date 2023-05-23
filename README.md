## Cтруктура проекта
```
.
├── checkpoints
│   # Здесь лежат модели в формате pickle. Рядом лежат txt с гиперпараметрами и лучшей метрикой на валидации
│   # В txt файлах присутствует опечатка, что метрика везде ROCAUC. Это не так, метрика зависит от датасета, задавалась отдельным параметром
│   ├── classification_Ames test   Mutagenicity_Mutagenicity_Ames Mutagenicity_automl.pkl
│   ├── classification_Ames test   Mutagenicity_Mutagenicity_Ames Mutagenicity_parameters.txt
│   ├── ...
│   ├── regression_Rat Subcutaneous LDLo_Acute Toxicity_rat_subcutaneous_LDLo_automl.pkl
│   └── regression_Rat Subcutaneous LDLo_Acute Toxicity_rat_subcutaneous_LDLo_parameters.txt
├── data
│   # Собранные датасеты
│   ├── B3DB
│   ├── cardiotox_hERG
│   ├── DeePred-BBB
│   ├── toxric
│   └── TrimNet
├── hackathon-data
│   # Данные от организаторов
│   ├── example_input.csv
│   ├── example_output_classification.csv
│   ├── example_output_regression.csv
│   └── example_train_dataset.csv
├── notebooks
│   # Jupyter notebooks для обучения и предсказания
│   ├── train.ipynb
│   └── predict.ipynb
├── scripts
│   # Скрипты для сбора и обработки данных и бенчмарков
│   ├── skin.py
│   ├── toxric_benchmark.py
│   ├── toxric.py
│   ├── tsv2csv.py
│   └── xlsx2csv.py
└── README.md
```

## Как запускать

1. Установите пакеты
```
pip install -r requirements.txt
```
2. В jupyter-notebook ``notebooks/predict.ipynb`` необходимо прописать нужные константы во второй ячейке. На выходе будет файл ``output.csv`` с двумя столбцами

## Наши датасеты и как они собирались
 - [x] [TOXRIC](https://toxric.bioinforai.tech/download) - с помощью скрипта ``scripts/toxric.py``
 - [x] [cardioTox hERG](https://github.com/Abdulk084/CardioTox/tree/master/data)
 - [x] [DeePred-BBB](https://www.frontiersin.org/articles/10.3389/fnins.2022.858126/full#supplementary-material) перевели из xlsx в csv с помощью ``scripts/xlsx2csv.py``
 - [x] [Blood-Brain Barrier Database (B3DB)](https://github.com/theochem/B3DB/tree/main/B3DB) перевели из tsv.gz в csv с помощью ``scripts/tsv2gz.py``
 - [x] [TrimNet](https://github.com/yvquanli/TrimNet/blob/master/trimnet_drug/data)
 - [x] [MED-Duluth](https://archive.epa.gov/med/med_archive_03/web/html/fathead_minnow.html) Скачали `` FATHEAD.ZIP``, затем внутри файл ``FHMDB.DBF`` перевели в csv с помощью [онлайн конвертора](https://onlineconvertfree.com/convert/dbf/)
 - [x] [SkinSensDB](https://cwtung.kmu.edu.tw/skinsensdb/download) Сделали KNNImputation на Human_data и перевели в csv ``scripts/skin.py``
 - [x] [CPDB](https://files.toxplanet.com/cpdb/cpdb.html) Выбраны мыши и крысы, среди них отобраны только самцы. Была выбраны категория "tba" для поиска проявлений туморогенеза в любом органе. После были отсеяны все соединения со значением TD50=10^31 мг/кг/д. Полученные данные представляют из себя TD50 для самцов выбранного вида животных.


## Бенчмарк

### 0-37 индексы это [ToxRic](https://toxric.bioinforai.tech/benchmark)

|   index | title                                   | benchmark_name                                    | metric   | baseline |   our_result |
|--------:|:----------------------------------------|:--------------------------------------------------|:---------|---------:|-------------:|
|       0 | Acute Toxicity                          | mouse_intraperitoneal_LD50                        | R2       |   0.7047 |            0.04259 |
|       1 | Acute Toxicity                          | guinea pig_intraperitoneal_LD50                   | R2       |   0.3162 |            0.09155 |
|       2 | Acute Toxicity                          | rat_intraperitoneal_LD50                          | R2       |   0.5481 |            0.2284 |
|       3 | Acute Toxicity                          | rabbit_intraperitoneal_LD50                       | R2       |  -0.0132 |            -0.1683 |
|       4 | Acute Toxicity                          | mouse_intraperitoneal_LDLo                        | R2       |   0.2724 |             0.1586 |
|       5 | Acute Toxicity                          | rat_intraperitoneal_LDLo                          | R2       |   0.3336 |            0.08039 |
|       6 | Acute Toxicity                          | mouse_intravenous_LD50                            | R2       |   0.7264 |            0.6026 |
|       7 | Acute Toxicity                          | guinea pig_intravenous_LD50                       | R2       |   0.5056 |            0.08449 |
|       8 | Acute Toxicity                          | mouse pig_intravenous_LD50                        | R2       |   0.4789 |            ***0.6026*** |
|       9 | Acute Toxicity                          | rabbit_intravenous_LD50                           | R2       |   0.6198 |            ***0.7907*** |
|      10 | Acute Toxicity                          | rat_intravenous_LD50                              | R2       |   0.6652 |            ***0.8142*** |
|      11 | Carcinogenicity                         | Carcinogenicity                                   | F1       |   0.6775 |            ***0.7291*** |
|      12 | Mutagenicity                            | Ames mutagenicity                                 | F1       |   0.8526 |            ***0.8527*** |
|      13 | Developmental and Reproductive Toxicity | Developmental toxicity                            | F1       |   0.9285 |            ***0.9577*** |
|      14 | Developmental and Reproductive Toxicity | Reproductive toxicity                             | F1       |   0.9377 |            ***0.9495*** |
|      15 | Hepatotoxicity                          | Hepatotoxicity                                    | F1       |   0.7516 |            ***0.7745*** |
|      16 | Cardiotoxicity                          | Cardiotoxicity1                                   | F1       |   0.5798 |            ***0.6774*** |
|      17 | Cardiotoxicity                          | Cardiotoxicity10                                  | F1       |   0.7964 |            ***0.8052*** |
|      18 | Cardiotoxicity                          | Cardiotoxicity30                                  | F1       |   0.9133 |            ***0.925***  |
|      19 | Cardiotoxicity                          | Cardiotoxicity5                                   | F1       |   0.7093 |            ***0.7299*** |
|      20 | Endocrine Disruption                    | NR-AR                                             | F1       |   0.6163 |            ***0.6444*** |
|      21 | Endocrine Disruption                    | NR-AR-LBD                                         | F1       |   0.6759 |            ***0.7342*** |
|      22 | Endocrine Disruption                    | NR-AhR                                            | F1       |   0.5649 |            ***0.6129*** |
|      23 | Endocrine Disruption                    | NR-Aromatase                                      | F1       |   0.4073 |            0.321  |
|      24 | Endocrine Disruption                    | NR-ER                                             | F1       |   0.4082 |            ***0.475***  |
|      25 | Endocrine Disruption                    | NR-ER-LBD                                         | F1       |   0.5346 |            ***0.56***   |
|      26 | Endocrine Disruption                    | NR-PPAR-gamma                                     | F1       |   0.2922 |            0.2105 |
|      27 | Endocrine Disruption                    | SR-ARE                                            | F1       |   0.5138 |            0.4358 |
|      28 | Endocrine Disruption                    | SR-ATAD5                                          | F1       |   0.2727 |            ***0.3076*** |
|      29 | Endocrine Disruption                    | SR-HSE                                            | F1       |   0.362  |            0.3364 |
|      30 | Endocrine Disruption                    | SR-MMP                                            | F1       |   0.6881 |            0.6301 |
|      31 | Endocrine Disruption                    | SR-p53                                            | F1       |   0.3323 |            ***0.3333*** |
|      32 | Irritation and Corrosion                | Eye irritation                                    | F1       |   0.9685 |            0.969 |
|      33 | Irritation and Corrosion                | Eye corrosion                                     | F1       |   0.9501 |            ***0.9573*** |
|      34 | Ecotoxicity                             | LC50DM                                            | R2       |   0.6099 |            0.3008 |
|      35 | Ecotoxicity                             | BCF                                               | R2       |   0.7569 |            0.1908 |
|      36 | Ecotoxicity                             | LC50                                              | R2       |   0.6951 |            0.2536 |
|      37 | Ecotoxicity                             | IGC50                                             | R2       |   0.7928 |            0.5622 |
|      38 | Skin Sensitization                      | Skin Sensitization                                | ROC_AUC  |   ?      |            0.7777 |
|      39 | Cardiotoxicity/hERG inhibition          | [cardiotox_hERG](https://github.com/Abdulk084/CardioTox/tree/master)                               | Accuracy  |   0.810      |            ***0.8515*** |
|      40 | Blood Brain Barrier Penetration         | [DeePred-BBB](https://www.frontiersin.org/articles/10.3389/fnins.2022.858126/full)                                       | ROC_AUC  |   0.987  |            ***0.9981*** |
|      41 | Blood Brain Barrier Penetration         | [B3DB](https://www.nature.com/articles/s41597-021-01069-5)                                              | ROC_AUC  |   ?      |            ***0.9615*** |
|      42 | Blood Brain Barrier Penetration         | [TrimNet](https://academic.oup.com/bib/article-abstract/22/4/bbaa266/5955940?redirectedFrom=fulltext)                                           | ROC_AUC  |   0.850       |            0.894  |
 

## Датасеты от организаторов
- [ ] PubChem
- [ ] ChemSpider
- [ ] ChemBL
- [ ] CompTox Chemicals Dashboard
- [ ] COMPTOX
- [x] TOXRIC
- [ ] Open Food Tox
- [ ] OpenTox
- [x] Acute Toxicity Test Database Query
- [x] Exploring ToxCost Data
- [x] TOX 21
- [ ] Comparative Toxicogenomics Database (CTD)
- [ ] ECOTOX
- [ ] European Chemicals Agency (ECHA)
- [ ] EMBL-EB I (European Bioinformatics Institute)
- [ ] Chemical Effects in Biological Systems (CEBS)
- [ ] UK Chemical Hazards Compendum
- [ ] Pharmaceuticals in the Environment, Information for Assessing Risk website
- [ ] Human Metabolome Database (HMDB)
- [ ] PA Integrated Risk Information System (IRIS)
- [ ] NORMAN Suspect List Exchange
- [x] SIDER
- [ ] The Carcinogenic Potency Database
- [ ] Chemical Carcinogenesis Research Information System
- [ ] Life Science Database Archive
- [ ] TOXICO DB
- [ ] KEGG : Kyoto Encyclopedia of Genes and Genomes
- [ ] RepDose
- [ ] Публикация описывающая построение CATMoS: Collaborative Acute