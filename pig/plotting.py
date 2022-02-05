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

def slide_plots():
    data = torch.load("results/full_scores.pt")
    data = ev.add_condition(data)
    data = score_points(data)
    data['pretraining'] = pd.Categorical(data.apply(ev.pretraining, axis=1),
                                         categories=['None', 'V', 'A', 'AV'])
    
    conditions = dict(jitter=[68, 206974],
                      static=[206974, 206978],
                      pretraining=[206974, 206975, 206976, 206977],
                      resolution=[206974, 206964])
    for condition, versions in conditions.items():
        for fragment_type in ['dialog', 'narration']:
            g = ggplot(data.query(f'fragment_type=="{fragment_type}" & version in {versions}'),
                       aes(x=condition, y='score')) + \
                       geom_boxplot() + \
                       facet_wrap('~metric') +\
                       ggtitle(fragment_type)
            ggsave(g, f"results/slides/{condition}_{fragment_type}.pdf")
    
    
