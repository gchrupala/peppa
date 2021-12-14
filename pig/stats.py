import pandas as pd
import statsmodels.formula.api as api
from sklearn.preprocessing import scale, StandardScaler
from sklearn.linear_model import RidgeCV
from plotnine import *
import torch
import numpy as np
from pig.grsa import VERSION

def sumcode(col):
    return (col * 2 - 1).astype(int)

def massage(dat, scaleall=False):
    dat['durationsum'] = dat['duration1'] + dat['duration2']
    keep = ['samespeaker', 'sameepisode', 'sametype', 'semsim',
            'durationdiff', 'durationsum', 'sim_1', 'sim_2']
    
    data = dat[keep].dropna().query("semsim != 0.0").assign(
        samespeaker  = lambda x: scale(x.samespeaker) if scaleall else sumcode(x.samespeaker),
        sameepisode = lambda x: scale(x.sameepisode) if scaleall else sumcode(x.sameepisode),
        sametype     = lambda x: scale(x.sametype) if scaleall else sumcode(x.sametype),
        semsim     = lambda x: scale(x.semsim),
        durationdiff = lambda x: scale(x.durationdiff),
        durationsum  = lambda x: scale(x.durationsum),
        sim_1 = lambda x: scale(x.sim_1),
        sim_2 = lambda x: scale(x.sim_2))
    return data



def rer(red, full):
    return (red - full) / red

def partial_r2(model, data):
    r2 = []
    mse_full = model.fit().mse_resid
    predictors = [ name for name in model.exog_names if name != 'Intercept' ]

    # drop intercept
    
    mse_red = model.from_formula(f"{model.endog_names} ~ {' + '.join(predictors)}",
                                 drop_cols=['Intercept'],
                                 data=data).fit().mse_resid
    r2.append(rer(mse_red, mse_full))
    for predictor in predictors:
        exog = ' + '.join([ name for name in predictors if name != predictor ])
        formula = f"{model.endog_names} ~ {exog}"
        mse_red = model.from_formula(formula, data).fit().mse_resid
        r2.append(rer(mse_red, mse_full))
    return pd.DataFrame(index=['Intercept']+predictors, data=dict(partial_r2=r2))
        

def plot_coef(fit, fragment_type, multiword):
    data = fit.reset_index().rename(
        columns={'index': 'Variable', 'Coef.': 'Coefficient', '[0.025': 'Lower', '0.975]': 'Upper'})
    g = ggplot(data, aes('Variable', 'Coefficient')) + \
        geom_hline(yintercept=0, color='gray', linetype='dashed') + \
        geom_errorbar(aes(color='Training', ymin='Lower', ymax='Upper', lwd=1, width=0.25)) + \
        geom_point(aes(color='Training')) + \
        coord_flip() 
    ggsave(g, f"results/grsa_{fragment_type}_{multiword}_coef.pdf")


def frameit(matrix, prefix="dim"):
    return pd.DataFrame(matrix, columns=[f"{prefix}{i}" for i in range(matrix.shape[1])])


def backprobes():
    for fragment_type in ['dialog', 'narration']:
        data = torch.load(f"data/out/words_{VERSION}_{fragment_type}.pt")
        backprobe(data['words']).to_csv(f"results/backprobe_{VERSION}_{fragment_type}.csv",
                                        index=False,
                                        header=True)
        
def backprobe(words):
    rows = []
    embedding_2 = frameit(scale(torch.stack([word.embedding_2 for word in words]).cpu().numpy()),
                          prefix="emb_2")
    embedding_1 = frameit(scale(torch.stack([word.embedding_1 for word in words]).cpu().numpy()),
                          prefix="emb_1")
    embedding_0 = frameit(scale(torch.stack([word.embedding_0 for word in words]).cpu().numpy()),
                          prefix="emb_0")
    semsim = frameit(torch.stack([word.semsim for word in words]).cpu().numpy(),
                    prefix="semsim")
    speaker = pd.get_dummies([word.speaker for word in words], prefix="speaker")
    episode = pd.get_dummies([word.episode for word in words], prefix="episode")
    duration = pd.DataFrame(dict(duration=[word.duration for word in words]))

    train_ix = np.random.choice(embedding_2.index, int(len(embedding_2.index)/2), replace=False)
    val_ix   = embedding_2.index[~embedding_2.index.isin(train_ix)]
    
    predictors = dict(semsim=semsim, speaker=speaker, episode=episode, duration=duration)
    for outname, y in [('embedding_2', embedding_2), ('embedding_1', embedding_1), ('embedding_0', embedding_0)]:
        X = pd.concat(list(predictors.values()), axis=1)
        full = ridge(X.loc[train_ix], y.loc[train_ix], X.loc[val_ix], y.loc[val_ix])
        rows.append(dict(var='NONE', outcome=outname, **full, rer=rer(full['mse'], full['mse'])))
        for name, X in ablate(predictors):
            red = ridge(X.loc[train_ix], y.loc[train_ix], X.loc[val_ix], y.loc[val_ix])
            rows.append(dict(var=name,
                             outcome=outname,
                             **red,
                             rer=rer(red['mse'], full['mse'])))
    return pd.DataFrame.from_records(rows)
        
def ridge_cv(X, y):
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_squared_error, r2_score
    model = make_pipeline(StandardScaler(),
                          RidgeCV(alphas=[ 10**n for n in range(-3, 11) ],
                                  fit_intercept=True, cv=None, scoring='neg_mean_squared_error',
                                  alpha_per_target=False
                          ))
    model.fit(X, y)
    return dict(mse= -model.steps[-1][1].best_score_,
                alpha=model.steps[-1][1].alpha_)


def ridge(X, y, X_val, y_val):
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_squared_error, r2_score
    model = make_pipeline(StandardScaler(),
                          RidgeCV(alphas=[ 10**n for n in range(-3, 11) ],
                                  fit_intercept=True, cv=None, scoring='neg_mean_squared_error',
                                  alpha_per_target=False
                          ))
    model.fit(X, y)
    pred = model.predict(X_val)
    return dict(mse=mean_squared_error(y_val, pred),
                alpha=model.steps[-1][1].alpha_,
                best_cv=-model.steps[-1][1].best_score_)

def ablate(variables):
    """Yield dataframe concatenating all variables, except for one each time."""
    for this in variables:
        yield this, pd.concat([ var for name, var in variables.items() if name != this ], axis=1) 

        
def main():
    # Load and process data

    training_mode = {0: "Untrained", 1: "Pre-trained", 2: "Fully-trained"}

    for multiword in ['multiword', 'word']:
        rawdata_d = pd.read_csv(f'data/out/pairwise_similarities_{multiword}_dialog.csv')
        data_d = massage(rawdata_d, scaleall=True)
        data_d.corr().to_csv(f"results/rsa_dialog_{multiword}_correlations.csv", index=True, header=True)
        data_d.corr().to_latex(float_format="%.2f", buf=f"results/rsa_dialog_{multiword}_correlations.tex")
    
        rawdata_n = pd.read_csv(f'data/out/pairwise_similarities_{multiword}_narration.csv')
        data_n = massage(rawdata_n, scaleall=True)
        data_ncor = data_n.drop("samespeaker", axis=1).corr()
        data_ncor.to_csv(f"results/rsa_narration_{multiword}_correlations.csv", index=True, header=True)
        data_ncor.to_latex(float_format="%.2f", buf=f"results/rsa_narration_{multiword}_correlations.tex")
        
        table_d = []
        for training in [1, 2]: 
            m_d = api.ols(formula = f'sim_{training} ~ semsim + durationdiff + durationsum + sametype + samespeaker + sameepisode', data=data_d)
            table = m_d.fit().summary2().tables[1]
            table['Training'] = training_mode[training]
            table_d.append(table)
        table_d = pd.concat(table_d, axis=0)
        table_d.to_csv(f"results/{multiword}_coef_d.csv", index=True, header=True)
    
        plot_coef(table_d, "dialog", multiword)


        table_n = []
        for training in [1, 2]: 
            m_n = api.ols(formula = f'sim_{training} ~ semsim + durationdiff + durationsum + sametype + sameepisode', data=data_n)
            table = m_n.fit().summary2().tables[1]
            table['Training'] = training_mode[training]
            table_n.append(table)
        table_n = pd.concat(table_n, axis=0)
        table_n.to_csv(f"results/{multiword}_coef_n.csv", index=True, header=True)
        plot_coef(table_n, "narration", multiword)
    
    



