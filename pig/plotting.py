from plotnine import *
import torch
import pandas as pd
import pig.evaluation as ev

def score_points(data):
    metrics = ['triplet_acc', 'recall_at_10_fixed', 'recall_at_10_jitter']
    rows = []
    for row in data:
        for metric in metrics:
            for score in row[metric]:
                point = { k:v for k, v in row.items()  if k not in metrics }
                if metric == 'triplet_acc':
                    point['score'] = score.item()
                else:
                    point['score'] = score.mean().item()
                point['metric'] = metric
                rows.append(point)
    return pd.DataFrame.from_records(rows)

def plots():
    data = torch.load("results/full_scores.pt")
    data = ev.add_condition(data)
    data = score_points(data)
    data['pretraining'] = pd.Categorical(data.apply(ev.pretraining, axis=1),
                                         categories=['None', 'V', 'A', 'AV'])
    conditions = dict(jitter=[333, 322],
                      static=[322, 326],
                      pretraining=[322, 323, 324, 325],
                      freeze_wav2vec=[322, 336])
    
    for condition, versions in conditions.items():
        g = ggplot(data.query(f'version in {versions}'),
                   aes(x=condition, y='score')) + \
                   geom_boxplot() + \
                   facet_wrap('~fragment_type + metric') 
        
        ggsave(g, f"results/ablations/{condition}.pdf")

    

