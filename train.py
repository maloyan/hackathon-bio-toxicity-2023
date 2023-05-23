import inspect
import math
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from flaml import AutoML
from mordred import Calculator, descriptors, error
from rdkit import Chem
from rdkit.Chem import Descriptors

warnings.filterwarnings("ignore")

DATASET_PATH = "/home/narek/kaggle/hackathon-bio/data"
CHECKPOINT_PATH = "/home/narek/kaggle/hackathon-bio/checkpoints"
LOG_PATH = "/home/narek/kaggle/hackathon-bio/logs"
TEST_DATASET_PATH = "/home/narek/kaggle/hackathon-bio/hackathon-data/"
TOP_N_FEATURES = 20
TIME = 120
TASK = 'regression'
DEBUG = True
NEG_LOG_MOL_PER_KG = True
METRICS = {
    "toxric": {
        "classification": "f1",
        "regression": "r2"
    },
    "cardiotox_hERG": {
        "classification": "accuracy",
        "regression": "r2"
    },
    "DeePred-BBB": {
        "classification": "roc_auc",
        "regression": "r2"
    },
    "B3DB": {
        "classification": "roc_auc",
        "regression": "r2"
    },
    "TrimNet": {
        "classification": "roc_auc",
        "regression": "r2"
    },
    "MED-Duluth": {
        "classification": "f1",
        "regression": "r2"
    },
    "SkinSensDB": {
        "classification": "roc_auc",
        "regression": "r2"
    },
    "CPDB": {
        "classification": "roc_auc",
        "regression": "r2"
    }
}

# Dictionary for mapping column names
COLUMNS_NAMES = {
    "toxric": {
        "Canonical SMILES": "smiles",
        "Toxicity Value": "target"
    },
    "cardiotox_hERG": {
        "smiles": "smiles",
        "ACTIVITY": "target"
    },
    "DeePred-BBB": {
        "Compounds": "smiles",
        "BBB-Class": "target"
    },
    "B3DB": {
        "SMILES": "smiles",
        "BBB+/BBB-": "target",
        #"logBB": "target"
    },
    "TrimNet": {
        "smiles": "smiles",
        "BBBP": "target"
    },
    "MED-Duluth": {
        "SMILES,C,100": "smiles",
        "LC50,N,12,5": "target"
    },
    "SkinSensDB": {
        "Canonical SMILES": "smiles",
        "Human_Data": "target"
    },
    "CPDB": {
        "smiles": "smiles",
        "target": "target"
    }
}

TARGETS = {
    "Developmental toxicity": [
        ("toxric", "Developmental and Reproductive Toxicity_Developmental Toxicity.csv", "classification"),
        ("toxric", "Developmental and Reproductive Toxicity_Reproductive Toxicity.csv", "classification"),
    ],
    "Skin Sensitization": [
        ("SkinSensDB", "data-20180205.csv", "classification")
    ],
    "Blood Brain Barrier Penetration": [
        ("DeePred-BBB", "Table 1.csv", "classification"),
        ("B3DB", "B3DB_classification_extended.csv", "classification"),
        #("B3DB", "B3DB_regression_extended.csv", "regression"),
        ("TrimNet", "bbbp.csv", "classification")
    ],
    "BBB-CHT mediated BBB permeation": [],
    "Hepatotoxicity": [
        ("toxric", "Hepatotoxicity_LTKB.csv", "classification")
    ],
    "Cardiotoxicity/hERG inhibition": [
        ("toxric", "Cardiotoxicity_Cardiotoxicity-1.csv", "classification"),
        ("toxric", "Cardiotoxicity_Cardiotoxicity-10.csv", "classification"),
        ("toxric", "Cardiotoxicity_Cardiotoxicity-30.csv", "classification"),
        ("toxric", "Cardiotoxicity_Cardiotoxicity-5.csv", "classification"),
        ("cardiotox_hERG", "train_validation_cardio_tox_data.csv", "classification"),
    ],
    "Carcinogenicity": [
        ("toxric", "Carcinogenicity_Carcinogenicity.csv", "classification")
    ],
    "Endocrine system disruption": [
        ("toxric", "Endocrine Disruption_NR-AR-LBD.csv", "classification"),
        ("toxric", "Endocrine Disruption_NR-AR.csv", "classification"),
        ("toxric", "Endocrine Disruption_NR-AhR.csv", "classification"),
        ("toxric", "Endocrine Disruption_NR-ER-LBD.csv", "classification"),
        ("toxric", "Endocrine Disruption_NR-ER.csv", "classification"),
        ("toxric", "Endocrine Disruption_NR-PPAR-gamma.csv", "classification"),
        ("toxric", "Endocrine Disruption_NR-aromatase.csv", "classification"),
        ("toxric", "Endocrine Disruption_SR-ARE.csv", "classification"),
        ("toxric", "Endocrine Disruption_SR-ATAD5.csv", "classification"),
        ("toxric", "Endocrine Disruption_SR-HSE.csv", "classification"),
        ("toxric", "Endocrine Disruption_SR-MMP.csv", "classification"),
        ("toxric", "Endocrine Disruption_SR-p53.csv", "classification"), #Перенести в Carcinogenicity
    ],
    "Eye Irritation": [
        ("toxric", "Irritation and Corrosion_Eye Irritation.csv", "classification")
    ],
    "Eye Corrosion": [
        ("toxric", "Irritation and Corrosion_Eye Corrosion.csv", "classification")
    ],
    "Mouse Intraperitoneal LD50": [
        ("toxric", "Acute Toxicity_mouse_intraperitoneal_LD50.csv", "regression")
    ],
    "Rat Intraperitoneal LD50": [
        ("toxric", "Acute Toxicity_rat_intraperitoneal_LD50.csv", "regression")
    ],
    "Rabbit Intraperitoneal LD50": [
        ("toxric", "Acute Toxicity_rabbit_intraperitoneal_LD50.csv", "regression")
    ],
    "Guinea Pig Intraperitoneal LD50": [
        ("toxric", "Acute Toxicity_guinea pig_intraperitoneal_LD50.csv", "regression")
    ],
    "Mouse Intraperitoneal LDLo": [
        ("toxric", "Acute Toxicity_mouse_intraperitoneal_LDLo.csv", "regression")
    ],
    "Rat Intraperitoneal LDLo": [
        ("toxric", "Acute Toxicity_rat_intraperitoneal_LDLo.csv", "regression")
    ],
    "Rabbit Intraperitoneal LDLo": [],
    "Guinea Pig Intraperitoneal LDLo": [],
    "Mouse Intravenous LD50": [
        ("toxric", "Acute Toxicity_mouse_intravenous_LD50.csv", "regression")
    ],
    "Rat Intravenous LD50": [
        ("toxric", "Acute Toxicity_rat_intravenous_LD50.csv", "regression")
    ],
    "Rabbit Intravenous LD50": [
        ("toxric", "Acute Toxicity_rabbit_intravenous_LD50.csv", "regression")
    ],
    "Guinea Pig Intravenous LD50": [
        ("toxric", "Acute Toxicity_guinea pig_intravenous_LD50.csv", "regression")
    ],
    "Mouse Intravenous LDLo": [
        ("toxric", "Acute Toxicity_mouse_intravenous_LDLo.csv", "regression")
    ],
    "Rat Intravenous LDLo": [
        ("toxric", "Acute Toxicity_rat_intravenous_LDLo.csv", "regression")
    ],
    "Rabbit Intravenous LDLo": [
        ("toxric", "Acute Toxicity_rabbit_intravenous_LDLo.csv", "regression")
    ],
    "Guinea Pig Intravenous LDLo": [
        ("toxric", "Acute Toxicity_guinea pig_intravenous_LDLo.csv", "regression")
    ],
    "Mouse Oral LD50": [
        ("toxric", "Acute Toxicity_mouse_oral_LD50.csv", "regression")
    ],
    "Rat Oral LD50": [
        ("toxric", "Acute Toxicity_rat_oral_LD50.csv", "regression")
    ],
    "Rabbit Oral LD50": [
        ("toxric", "Acute Toxicity_rabbit_oral_LD50.csv", "regression")
    ],
    "Guinea Pig Oral LD50": [
        ("toxric", "Acute Toxicity_guinea pig_oral_LD50.csv", "regression")
    ],
    "Mouse Oral LDLo": [
        ("toxric", "Acute Toxicity_mouse_oral_LDLo.csv", "regression")
    ],
    "Rat Oral LDLo": [
        ("toxric", "Acute Toxicity_rat_oral_LDLo.csv", "regression")
    ],
    "Rabbit Oral LDLo": [
        ("toxric", "Acute Toxicity_rabbit_oral_LDLo.csv", "regression")
    ],
    "Guinea Pig Oral LDLo": [],
    "Mouse Subcutaneous LD50": [
        ("toxric", "Acute Toxicity_mouse_subcutaneous_LD50.csv", "regression")
    ],
    "Rat Subcutaneous LD50": [
        ("toxric", "Acute Toxicity_rat_subcutaneous_LD50.csv", "regression")
    ],
    "Rabbit Subcutaneous LD50": [
        ("toxric", "Acute Toxicity_rabbit_subcutaneous_LD50.csv", "regression")
    ],
    "Guinea Pig Subcutaneous LD50": [
        ("toxric", "Acute Toxicity_guinea pig_subcutaneous_LD50.csv", "regression")
    ],
    "Mouse Subcutaneous LDLo": [
        ("toxric", "Acute Toxicity_mouse_subcutaneous_LDLo.csv", "regression")
    ],
    "Rat Subcutaneous LDLo": [
        ("toxric", "Acute Toxicity_rat_subcutaneous_LDLo.csv", "regression")
    ],
    "Rabbit Subcutaneous LDLo": [
        ("toxric", "Acute Toxicity_rabbit_subcutaneous_LDLo.csv", "regression")
    ],
    "Guinea Pig Subcutaneous LDLo": [
        ("toxric", "Acute Toxicity_guinea pig_subcutaneous_LDLo.csv", "regression")
    ],
    "Mouse Skin LD50": [
        ("toxric", "Acute Toxicity_mouse_skin_LD50.csv", "regression")
    ],
    "Rat Skin LD50": [
        ("toxric", "Acute Toxicity_rat_skin_LD50.csv", "regression")
    ],
    "Rabbit Skin LD50": [
        ("toxric", "Acute Toxicity_rabbit_skin_LD50.csv", "regression")
    ],
    "Guinea Pig Skin LD50": [
        ("toxric", "Acute Toxicity_guinea pig_skin_LD50.csv", "regression")
    ],
    "Mouse Skin LDLo": [],
    "Rat Skin LDLo": [],
    "Rabbit Skin LDLo": [
        ("toxric", "Acute Toxicity_rabbit_skin_LDLo.csv", "regression")
    ],
    "Guinea Pig Skin LDLo": [],
    "Ames test / Mutagenicity": [
        ("toxric", "Mutagenicity_Ames Mutagenicity.csv", "classification")
    ],
    "Bioconcentration factor": [
        ("toxric", "Ecotoxicity_BCF.csv", "regression")
    ],
    "40 hour Tetrahymena pyriformis IGC50": [
        ("toxric", "Ecotoxicity_IGC50.csv", "regression")
    ],
    "48 hour Daphnia magna LC50": [
        ("toxric", "Ecotoxicity_LC50DM.csv", "regression")
    ],
    "96 hour Fathead Minnow LC50": [
        ("MED-Duluth", "fhmdb.csv", "regression")
    ],
    "Mouse Carcinogenic potency TD50": [
        ("CPDB", "TD50_m.csv", "regression")
    ],
    "Rat Carcinogenic potency TD50": [
        ("CPDB", "TD50_r.csv", "regression")
    ],
    "Mouse NOAEL": [],
    "Rat NOAEL": [],
    "Mouse LOAEL": [],
    "Rat LOAEL": [],
    "Класс опасности по острой токсичности для водной среды / при попадании на кожу / при вдыхании / при проглатывании": [],
    "Класс опасности по хронической токсичности для водной среды": [],
    "Класс опасности по репродуктивной токсичности": [],
    "Классы опасности по раздражению глаз/кожи": [],
    "Класс опасности по мутагенности": [],
    "Класс опасности по канцерогенности": [],
    "Класс опасности по избирательной токсичности при однократном/многократном введении": [],
}

# Calculate all molecular descriptors from RDKit and mordred
def calculate_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}

    # Get all descriptor functions from RDKit Descriptors module
    rdkit_descriptor_functions = [
        descriptor for descriptor in dir(Descriptors)
        if callable(getattr(Descriptors, descriptor))
        and not descriptor.startswith('_')
        and len(inspect.signature(getattr(Descriptors, descriptor)).parameters) == 1
    ]

    # Calculate RDKit descriptor values for the given molecule
    rdkit_descriptor_values = [
        getattr(Descriptors, descriptor)(mol) for descriptor in rdkit_descriptor_functions
    ]

    # Create a descriptor calculator with all available Mordred descriptors
    mordred_calc = Calculator(descriptors, ignore_3D=True)

    try:
        # Calculate Mordred descriptor values for the given molecule
        mordred_descriptor_values = mordred_calc(mol)
    except ValueError as e:
        return {"error": f"{e}: {smiles}"}

    # Replace Mordred error objects with None
    mordred_descriptor_values = [value if not isinstance(value, error.MissingValueBase) else None for value in mordred_descriptor_values]

    # Combine RDKit and Mordred descriptor names and values
    descriptor_names = rdkit_descriptor_functions + [str(d) for d in mordred_calc.descriptors]
    descriptor_values = rdkit_descriptor_values + list(mordred_descriptor_values)

    # Return descriptor values with names as a dictionary
    return dict(zip(descriptor_names, descriptor_values))

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def get_molecular_weight(smiles):
    """
    Calculate the molecular weight of a compound given its SMILES notation.

    :param smiles: The SMILES notation of the compound.
    :return: The molecular weight of the compound.
    """
    molecule = Chem.MolFromSmiles(smiles)
    molecular_weight = Descriptors.MolWt(molecule)
    return molecular_weight

def convert_mg_per_kg_to_neg_log_mol_per_kg(mg_per_kg, molecular_weight):
    """
    Convert mg/kg to negative log mol/kg.

    :param mg_per_kg: The concentration in mg/kg.
    :param molecular_weight: The molecular weight of the substance in g/mol.
    :return: The negative log mol/kg concentration.
    """
    g_per_kg = mg_per_kg / 1000
    mol_per_kg = g_per_kg / molecular_weight
    neg_log_mol_per_kg = -math.log10(mol_per_kg)

    return neg_log_mol_per_kg

def calculate_neg_log_mol_per_kg(row):
    smiles = row['smiles']
    mg_per_kg = row['target']
    molecular_weight = get_molecular_weight(smiles)
    neg_log_mol_per_kg = convert_mg_per_kg_to_neg_log_mol_per_kg(mg_per_kg, molecular_weight)
    return neg_log_mol_per_kg


def explain_model(model, top_features, file):
    print('Best ML leaner:', model.best_estimator, file=file)
    print('Best hyperparmeter config:', model.best_config, file=file)
    print('Best ROC_AUC on validation data: {0:.4g}'.format(1-model.best_loss), file=file)
    print('Training duration of best run: {0:.4g} s'.format(model.best_config_train_time), file=file)

    TOP_N_FEATURES = 20
    # Get the feature importances from the AutoML model
    feature_importances = model.feature_importances_

    # Create a list of tuples that pairs each feature name with its importance
    feature_tuples = zip(model.feature_names_in_, feature_importances)

    # Sort the tuples by importance in descending order
    sorted_features = sorted(feature_tuples, key=lambda x: x[1], reverse=True)

    # Extract the sorted feature names and importances into separate lists
    sorted_names, sorted_importances = zip(*sorted_features)
    print("Top features:", file=file)
    print(sorted_names[:TOP_N_FEATURES], file=file)
    print(sorted_importances[:TOP_N_FEATURES], file=file)
    # Plot the sorted feature importances
    plt.barh(sorted_names[:TOP_N_FEATURES], sorted_importances[:TOP_N_FEATURES])

benchmarks = []
results = []
metric_names = []
for target in TARGETS:
    print(target)
    if len(TARGETS[target]) != 0:
        for dataset_name, file_path, task in TARGETS[target]:
            if task == TASK:
                filename = file_path.split('.')[0]
                df = pd.read_csv(os.path.join(DATASET_PATH, dataset_name, file_path))
                if df.shape[0] != 0:
                    # Rename column
                    df.rename(columns=COLUMNS_NAMES[dataset_name], inplace=True)
                    df = df[["smiles", "target"]]

                    df = df.drop_duplicates(subset=['smiles'])
                    df.dropna(inplace=True)
                    df.reset_index(inplace=True, drop=True)

                    df['is_valid_smiles'] = df.smiles.apply(is_valid_smiles)
                    print(f"Something is wrong with {df[df.is_valid_smiles == False].shape[0]} number of SMILES")

                    df = df[df.is_valid_smiles == True]
                    df.drop(columns=['is_valid_smiles'], inplace=True)

                    # For each df["Canonical SMILES"] calculate all descriptors
                    # using calculate_all_descriptors function which returns a dictionary
                    # with descriptor names as keys and descriptor values as values
                    X_train = df["smiles"].apply(calculate_all_descriptors)

                    # Convert dictionary to a dataframe, add SMILES column and Toxicity Value column
                    X_train = pd.DataFrame(X_train.tolist())
                    y_train = df["target"].values

                    target = target.replace('/', ' ')
                    # Initialize an AutoML instance
                    automl = AutoML()
                    # Specify automl goal and constraint
                    automl_settings = {
                        "time_budget": TIME,  # in seconds
                        "metric": METRICS[dataset_name][TASK],
                        "task": TASK,
                        "split_ratio": 0.2,
                        "n_splits": 5,
                        #"split_type": "stratified",
                        #"ensemble": True,
                        "verbose": 2,
                        "seed": 42,
                        "log_file_name": f"{LOG_PATH}/{TASK}_{target}_{filename}.log",
                        "early_stop":True
                    }
                    automl.fit(
                        X_train=X_train, 
                        y_train=y_train,
                        **automl_settings
                    )
                    # your code that produces output
                    with open(f'{CHECKPOINT_PATH}/{TASK}_{target}_{filename}_parameters.txt', 'w') as f:
                        explain_model(automl, TOP_N_FEATURES, file=f)  # Python 3.x f.write(cap_out.stdout)
                    f.close()

                    # Save best model
                    with open(f'{CHECKPOINT_PATH}/{TASK}_{target}_{filename}_automl.pkl', 'wb') as f:
                        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

                    benchmarks.append(f"{TASK}_{target}_{filename}")

                    # Мы используем метрику оценки качества классификации как лосс, поэтому берем 1 - best_loss
                    # Так как лосс чем ближе к 0 тем лучше, а метрика наоборот наоборот
                    # Подбробнее в документации FLAML https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML#customize-automlfit
                    results.append(1 - automl.best_loss)
                    metric_names.append(METRICS[dataset_name][TASK])

df_final_res = pd.DataFrame()
df_final_res['benchmark_name'] = benchmarks
df_final_res['value'] = results
df_final_res['metric'] = metric_names

df_final_res.to_csv(f"{TASK}_benchmark.csv", index=None)
