#!/usr/bin/env python3
import pandas as pd

lex = pd.read_csv('jianyu_prediction_pred_lex.csv')
lem = pd.read_csv('jianyu_prediction_pred_lem.csv')

def pick(cols, keys):
    for c in cols:
        s = str(c).strip().lower()
        for k in keys:
            if s.startswith(k):
                return c
    return cols[0] if cols else None

lex_col = pick(list(lex.columns), ['λex_pred', 'lex_pred'])
lem_col = pick(list(lem.columns), ['λem_pred', 'lem_pred'])
df = lex[[lex_col]].join(lem[[lem_col]], how='inner')
df.columns = ['ex', 'em']
print(f"[post] processbefore ex>em items数: {(df['ex']>df['em']).sum()} / {len(df)}")


