# -*- coding: utf-8 -*-
"""四档防泄露协议 × (本项目前20配置 + 李宏伟2套特征方案) 同口径对比。

协议递进:
  R1 完全不防泄露  : resubstitution, 全量训练 + 全量测试(无任何切分)
  R2 只 K 折        : 普通 KFold(n_splits=5, shuffle) 按行随机, 同曲线跨训练/测试
  R3 只渣样级防泄露 : GroupKFold 5x2, group = 日期_炉号 (项目 SEED=42 同款折生成)
  R4 完全防泄露     : 严格时间外推, 训练 <=2026-07-31, 测试其后 327 点
                      (因 group 含日期, 时间外推天然同时满足渣样级不泄漏)

为便于跨协议比较, R1-R3 额外在 R4 的测试期(2026-08 之后)同一批点上再算一次指标。
所有模型定义忠实取自 analysis_v2/interval_pipeline.py, SEED=42。
"""
import numpy as np, pandas as pd, io, json, time, warnings
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostRegressor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import OUT, PLANT_CSV, GPVISC_CSV  # noqa: E402


warnings.filterwarnings('ignore')
D = str(Path(__file__).resolve().parents[1])
SEED, N_REPEAT, N_SPLITS = 42, 2, 5
INV_T_REF = 1000.0 / (1450 + 273.15)

data = pd.read_csv(PLANT_CSV, encoding='utf-8-sig')
SUB = data.dropna(subset=['FeO', 'S']).copy()
LATE = data['date'].values > '2026-07-31'
print('全量 %d 点 / %d 组 | F4 子集 %d 点 | 末期(>2026-07-31) %d 点' % (
    len(data), data['group'].nunique(), len(SUB), LATE.sum()))

SCHEMES = {
    'F1质量分数': ['SiO2', 'Al2O3', 'CaO', 'MgO', 'invT'],
    'F2质量+派生': ['SiO2', 'Al2O3', 'CaO', 'MgO', 'R', 'R4', 'MgO_Al2O3', 'invT'],
    'F3摩尔+碱度': ['x_SiO2', 'x_Al2O3', 'x_CaO', 'x_MgO', 'Lambda', 'invT'],
    'F4全成分': ['SiO2', 'Al2O3', 'CaO', 'MgO', 'R', 'R4', 'MgO_Al2O3', 'FeO', 'S', 'invT'],
}


def ttr(m):
    return TransformedTargetRegressor(regressor=m, transformer=StandardScaler())


def mk_model(name, seed):
    return {
        'Ridge': ttr(Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))])),
        'PLS': ttr(Pipeline([('sc', StandardScaler()), ('m', PLSRegression(n_components=3))])),
        'KNN': Pipeline([('sc', StandardScaler()), ('m', KNeighborsRegressor(n_neighbors=7, weights='distance'))]),
        'RF': RandomForestRegressor(n_estimators=400, max_features=0.5, random_state=seed, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=400, max_features=0.5, random_state=seed, n_jobs=-1),
        'HistGB': HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=6, random_state=seed),
        'XGBoost': xgb.XGBRegressor(n_estimators=400, learning_rate=0.06, max_depth=6, subsample=0.8,
                                    colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                                    n_jobs=-1, verbosity=0),
        'LightGBM': lgb.LGBMRegressor(n_estimators=400, learning_rate=0.06, num_leaves=31, subsample=0.8,
                                      colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=-1),
        'CatBoost': CatBoostRegressor(iterations=400, learning_rate=0.06, depth=6, random_seed=seed,
                                      verbose=0, allow_writing_files=False),
    }[name]


# ---------- 特征与模型清单 ----------
CONFIGS = [
    ('ExtraTrees', 'F3摩尔+碱度'), ('RF', 'F3摩尔+碱度'), ('Ridge', 'F2质量+派生'),
    ('PLS', 'F2质量+派生'), ('Ridge', 'F1质量分数'), ('PLS', 'F1质量分数'),
    ('B1-Arrhenius', '机理基线'), ('XGBoost', 'F3摩尔+碱度'), ('ExtraTrees', 'F2质量+派生'),
    ('ExtraTrees', 'F1质量分数'), ('CatBoost', 'F3摩尔+碱度'), ('HistGB', 'F3摩尔+碱度'),
    ('ExtraTrees', 'F4全成分'), ('KNN', 'F3摩尔+碱度'), ('H2-活化能映射', '创新混合'),
    ('Ridge', 'F3摩尔+碱度'), ('PLS', 'F3摩尔+碱度'), ('LightGBM', 'F3摩尔+碱度'),
    ('KNN', 'F2质量+派生'), ('RF', 'F2质量+派生'),
]

# ---------- 李宏伟结构特征 ----------
def lhw_features(df):
    S = df[['SiO2', 'Al2O3', 'CaO', 'MgO']].copy()
    S = S.div(S.sum(axis=1), axis=0) * 100.0
    nCaO, nSiO2 = S['CaO'] / 56, S['SiO2'] / 64
    nAl2O3, nMgO = S['Al2O3'] / 102, S['MgO'] / 40
    den = nSiO2 + 2 * nAl2O3
    nCa_Si = 2 * (nCaO - nAl2O3) * nSiO2 / den
    nMg_Si = 2 * nMgO * nSiO2 / den
    nCa_AlCa = 2 * nCaO * 2 * nAl2O3 / den
    nMg_AlCa = 2 * (nCaO - nAl2O3) * 2 * nAl2O3 / den
    nSi = 2 * nSiO2 - nCa_Si / 2 - nMg_Si / 2
    nCa_Al = 4 * nAl2O3 - nCa_AlCa / 2 - nMg_AlCa / 2
    invT = 1000.0 / (df['T_C'].values + 273.15)
    return {
        'LHW-4成分+T': np.column_stack([S['SiO2'], S['Al2O3'], S['CaO'], S['MgO'], invT]),
        'LHW-6氧+T': np.column_stack([nCa_Si, nMg_Si, nCa_AlCa, nMg_AlCa, nSi, nCa_Al, invT]),
    }


def lhw_predict(Xtr, ytr, Xte, seed):
    sc = StandardScaler().fit(Xtr)
    preds = []
    for h in range(4, 14):
        m = MLPRegressor(hidden_layer_sizes=(h,), activation='logistic', solver='lbfgs',
                         max_iter=30000, random_state=SEED + h)
        m.fit(sc.transform(Xtr), ytr)
        preds.append(m.predict(sc.transform(Xte)))
    return np.mean(preds, axis=0)


# ---------- 通用预测器 ----------
def predict_one(mname, scheme, dset, tr, te, seed):
    """返回 (pred_array_on_te, )"""
    yv = dset['ln_eta'].values
    if mname == 'B1-Arrhenius':
        Xb = np.column_stack([np.ones(len(tr)), dset['invT'].values[tr]])
        cb, *_ = np.linalg.lstsq(Xb, yv[tr], rcond=None)
        return cb[0] + cb[1] * dset['invT'].values[te]
    if mname == 'H2-活化能映射':
        feats = SCHEMES['F2质量+派生']
        feats_c = [c for c in feats if c != 'invT']
        trd = dset.iloc[tr]
        ka_rows = []
        for gk, g in trd.groupby('group'):
            liq = g.sort_values('T_C', ascending=False)
            if len(liq) >= 4:
                k, a = np.polyfit(liq['invT'] - INV_T_REF, liq['ln_eta'], 1)
                ka_rows.append((gk, k, a))
        ka = pd.DataFrame(ka_rows, columns=['group', 'k', 'a1450']).set_index('group')
        info_tr = dset.drop_duplicates('group').set_index('group').loc[ka.index]
        mk = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                               random_state=seed, n_jobs=-1, verbosity=-1)
        ma = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                               random_state=seed + 1, n_jobs=-1, verbosity=-1)
        mk.fit(info_tr[feats_c], ka['k']); ma.fit(info_tr[feats_c], ka['a1450'])
        ted = dset.iloc[te]
        info_te = dset.drop_duplicates('group').set_index('group').loc[pd.unique(ted['group'])]
        kk = dict(zip(info_te.index, mk.predict(info_te[feats_c])))
        aa = dict(zip(info_te.index, ma.predict(info_te[feats_c])))
        return np.array([aa[g] + kk[g] * (it - INV_T_REF)
                         for g, it in zip(ted['group'], ted['invT'])])
    X = dset[SCHEMES[scheme]].values
    m = mk_model(mname, seed)
    m.fit(X[tr], yv[tr])
    return np.asarray(m.predict(X[te])).ravel()


# ---------- 协议定义 ----------
def splits_R1(dset):
    n = len(dset)
    return [('R1 不防泄露 (resubstitution)', np.arange(n), np.arange(n))]


def splits_R2(dset):
    out = []
    for i, (tr, te) in enumerate(KFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(dset)))):
        out.append(('R2 只K折 (按行随机)', tr, te))
    return out


def group_folds(groups, n_splits=N_SPLITS, n_repeat=N_REPEAT, seed=SEED):
    uniq = pd.unique(groups)
    for rep in range(n_repeat):
        perm = np.random.RandomState(seed + rep).permutation(uniq)
        rm = {g: i for i, g in enumerate(perm)}
        gi = np.array([rm[g] for g in groups])
        for tr, te in GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=gi):
            yield tr, te


def splits_R3(dset):
    return [('R3 渣样级防泄露 (GroupKFold 5x2)', tr, te)
            for tr, te in group_folds(dset['group'].values)]


def splits_R4(dset):
    late = dset['date'].values > '2026-07-31'
    return [('R4 完全防泄露 (时间外推)', np.where(~late)[0], np.where(late)[0])]


PROTOCOLS = [('R1', splits_R1), ('R2', splits_R2), ('R3', splits_R3), ('R4', splits_R4)]


# ---------- 指标 ----------
def metrics(yv, pv):
    if len(yv) == 0:
        return dict(rmse_ln=np.nan, R2=np.nan, rmse_Pa_s=np.nan, mae_Pa_s=np.nan, rel_med=np.nan)
    return dict(rmse_ln=float(np.sqrt(np.mean((yv - pv) ** 2))),
                R2=float(1 - np.sum((yv - pv) ** 2) / np.sum((yv - yv.mean()) ** 2)),
                rmse_Pa_s=float(np.sqrt(np.mean((np.exp(yv) - np.exp(pv)) ** 2))),
                mae_Pa_s=float(np.mean(np.abs(np.exp(yv) - np.exp(pv)))),
                rel_med=float(np.median(np.abs(np.exp(pv) - np.exp(yv)) / np.exp(yv)) * 100))


rows = []
t0 = time.time()

ALL = [(m, s, False) for m, s in CONFIGS] + [(k, '外部方法', True) for k in ['LHW-4成分+T', 'LHW-6氧+T']]

for mname, scheme, is_lhw in ALL:
    dset = SUB if scheme == 'F4全成分' else data
    yv = dset['ln_eta'].values
    late_mask = dset['date'].values > '2026-07-31'
    for pname, fn in PROTOCOLS:
        oof = np.full(len(dset), np.nan)
        fold_rmses = []
        for k, (_lab, tr, te) in enumerate(fn(dset)):
            if is_lhw:
                feats = lhw_features(dset)
                key = mname
                p = lhw_predict(feats[key][tr], yv[tr], feats[key][te], SEED + k)
            else:
                p = predict_one(mname, scheme, dset, tr, te, SEED + k)
            oof[te] = p
            fold_rmses.append(np.sqrt(mean_squared_error(yv[te], p)))
        ok = ~np.isnan(oof)
        m_all = metrics(yv[ok], oof[ok])
        m_late = (metrics(yv[late_mask], oof[late_mask])
                  if (late_mask & ok).sum() else dict(rmse_ln=np.nan, R2=np.nan, rmse_Pa_s=np.nan))
        rows.append(dict(模型=mname, 方案=scheme, 来源='外部课题组' if is_lhw else '本项目',
                         协议=pname, 折数=len(fn(dset)), 测试点=int(ok.sum()),
                         **m_all, late_rmse_ln=m_late.get('rmse_ln', np.nan),
                         late_R2=m_late.get('R2', np.nan), late_rmse_Pa_s=m_late.get('rmse_Pa_s', np.nan),
                         折间std=float(np.std(fold_rmses))))
    print('  %-16s %-12s 完成  R1=%.4f R2=%.4f R3=%.4f R4=%.4f  (%.0fs)' % (
        mname, scheme,
        [r for r in rows if r['模型'] == mname and r['方案'] == scheme][0]['rmse_ln'],
        [r for r in rows if r['模型'] == mname and r['方案'] == scheme][1]['rmse_ln'],
        [r for r in rows if r['模型'] == mname and r['方案'] == scheme][2]['rmse_ln'],
        [r for r in rows if r['模型'] == mname and r['方案'] == scheme][3]['rmse_ln'],
        time.time() - t0), flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / '_G4_leakage_ladder.csv', index=False, encoding='utf-8-sig')
print('\n全部完成，用时 %.0f s，已落盘 %s' % (time.time() - t0, '_G4_leakage_ladder.csv'))
