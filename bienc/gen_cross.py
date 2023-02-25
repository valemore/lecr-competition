def gen_cross_df(distances, indices, corr_df, e2i):
    i2e = {idx: content_id for content_id, idx in e2i.items()}
    all_cands = []
    for pred_idxs in indices:
        cands = [i2e[pred_idx] for pred_idx in pred_idxs]
        all_cands.append(" ".join(cands))
    gen_df = corr_df.copy()
    gen_df["cands"] = all_cands
    return gen_df
