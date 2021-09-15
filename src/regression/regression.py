from typing import List

from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import lang2vec.lang2vec as l2v
import pycountry
import numpy as np
from utils import get_script, get_lang_family
np.set_printoptions(precision=3)

TARGET_LANGS = "en ru zh ar hi te ur mr".split()  # Example target languages

iso2iso3 = {lang: pycountry.languages.get(alpha_2=lang).alpha_3 for lang in TARGET_LANGS}
langs = [iso2iso3[lang] for lang in TARGET_LANGS]

# Typological features
lang_features_syntax = l2v.get_features(langs, "syntax_knn", header=True)
lang_features_phon = l2v.get_features(langs, "phonology_knn", header=True)
lang_features_id = l2v.get_features(langs, "id", header=True)

SCRIPT = get_script()
FAMILY = get_lang_family()
PRETRAIN = {k: k for k in SCRIPT.keys()}
acc = {}

X = []
Y = []

# Read the data
pretrain_langs = ["en", "en_ru_zh_ar", "en_ru_zh_ar_hi"]  # Example source languages


def add_feature(feature: dict,
                x: List[int],
                pretrain_lang_used: List[str]):
    flag = False
    for pre in pretrain_lang_used:
        if feature[pre] == feature[language]:
            x.append(1)
            flag = True
            break
    if not flag:
        x.append(0)


for pretrain_lang in pretrain_langs:
    pretrain_lang_used = pretrain_lang.split("_")
    print(pretrain_lang)
    acc_sum = 0
    num_lang = 0

    acc_sum_seen = 0
    num_lang_seen = 0
    language = ""
    accuracy = 0


    with open(f"src/regression/example_output/"
              f"{pretrain_lang}/test_results.txt") as f:

        for line in f:
            if line.startswith("======="):
                x = []
                if not language: continue
                if not language in TARGET_LANGS: continue
                if language in "en_ru_zh_ar_hi".split("_"): continue
                print(language, accuracy)
                acc[language] = accuracy

                add_feature(SCRIPT, x, pretrain_lang_used)
                add_feature(FAMILY, x, pretrain_lang_used)

                max_cos_sim_syntax = -10.0
                iso3_target = iso2iso3[language]
                for l in pretrain_lang_used:
                    iso3_pre = iso2iso3[l]
                    cos_sim_syntax = 1 - cosine(lang_features_syntax[iso3_target],
                                                lang_features_syntax[iso3_pre])
                    max_cos_sim_syntax = max(cos_sim_syntax, max_cos_sim_syntax)
                x.append(max_cos_sim_syntax)

                max_cos_sim_phon = -10.0
                iso3_target = iso2iso3[language]
                for l in pretrain_lang_used:
                    iso3_pre = iso2iso3[l]
                    cos_sim_phon = 1 - cosine(lang_features_phon[iso3_target],
                                              lang_features_phon[iso3_pre])
                    max_cos_sim_phon = max(cos_sim_phon, max_cos_sim_phon)
                x.append(max_cos_sim_phon)

                x.append(len(pretrain_lang_used))
                X.append(x)
                Y.append(accuracy)

                language = ""
                accuracy = 0

            if line.startswith("language="):
                language = line.strip()[-2:]
            if line.startswith("accuracy"):
                accuracy = float(line.strip().split(' ')[-1])

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X = np.array(X)
Y = np.array(Y)

X2 = sm.add_constant(X)
mod = sm.OLS(Y, X2)
fii = mod.fit()
print(fii.summary())
p_values = fii.summary2().tables[1]['P>|t|']
coef = fii.summary2().tables[1]['Coef.']

# Numbers related to correlation and colinearity
# print("--correlation matrix--")
# print(np.corrcoef(X.T))
# corr = np.corrcoef(X.T)
# print(np.mean(fii.resid))
# print(np.std(fii.resid))
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# print("Variance inflation factor (VIF) to quantify the effect of collinearity...")
# print("If larger than 5, then the collinearity is problematic for that feature...")
# for k, vif in enumerate([variance_inflation_factor(X, j) for j in range(X.shape[1])]):
#     print(k, "%.4f" % vif)
