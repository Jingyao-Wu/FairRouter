import torch
ENVIRONMENTS = ('clean', 'text_truncate_050', 'text_mask_mixed_030', 'text_span_delete_030', 'graph_drop_030', 'graph_ego_mixed_050')

def targets(gnn_pred, llm_pred, labels):
    g, l, y = [torch.as_tensor(x, dtype=torch.long).reshape(-1) for x in (gnn_pred, llm_pred, labels)]
    if not g.shape == l.shape == y.shape or bool((torch.cat((g, l, y)) < 0).any()):
        raise ValueError('invalid predictions/labels')
    agree = g == l
    gc, lc = (g == y, l == y)
    dis = ~agree
    decisive = dis & (gc | lc)
    missing = torch.full_like(g, -1)
    return dict(agreement_mask=agree, disagreement_mask=dis, agreement_target=torch.where(agree, gc.long(), missing), trust_target=torch.where(dis, (gc | lc).long(), missing), preference_target=torch.where(decisive, gc.long(), missing), preference_mask=decisive, state=torch.where(dis, torch.where(gc, 0, torch.where(lc, 1, 2)), missing))

def organize(dataset, environments, *, gold_ids, gold_targets, num_nodes):
    gold_ids = torch.as_tensor(gold_ids, dtype=torch.long)
    gold_targets = torch.as_tensor(gold_targets, dtype=torch.long)
    if gold_ids.ndim != 1 or gold_ids.shape != gold_targets.shape or gold_ids.unique().numel() != len(gold_ids):
        raise ValueError('invalid gold support')
    gold = {int(n): int(y) for n, y in zip(gold_ids, gold_targets)}
    seen = set()
    pieces = []
    tabs = []
    gs = []
    ls = []
    for env in environments:
        name = env['environment']
        ids = torch.as_tensor(env['node_ids'], dtype=torch.long)
        if name not in ENVIRONMENTS:
            raise ValueError('unregistered environment')
        keys = [(name, int(n)) for n in ids]
        if len(set(keys)) != len(keys) or seen.intersection(keys):
            raise ValueError('duplicate node/environment')
        seen.update(keys)
        if not set(ids.tolist()) <= set(gold) or bool(((ids < 0) | (ids >= num_nodes)).any()):
            raise ValueError('every augmented parent must be a gold node')
        n = len(ids)
        pg = env['gnn_prob']
        pl = env['llm_prob']
        if pg.shape != pl.shape or pg.ndim != 2 or pg.shape[0] != n or (pg.shape[1] < 2):
            raise ValueError('invalid expert probability shape')
        for p in (pg, pl):
            if not bool(torch.isfinite(p).all()) or bool((p < 0).any()) or (not torch.allclose(p.sum(1), torch.ones(n), atol=1e-05, rtol=1e-05)):
                raise ValueError('invalid expert probabilities')
        for k, w in [('tab70', 70), ('g', 128), ('l', 4096)]:
            if env[k].shape != (n, w) or not bool(torch.isfinite(env[k]).all()):
                raise ValueError('invalid ' + k)
        g = pg.argmax(1)
        l = pl.argmax(1)
        y = torch.tensor([gold[int(i)] for i in ids])
        if bool((y >= pg.shape[1]).any()):
            raise ValueError('gold class outside probability width')
        env_id = ENVIRONMENTS.index(name)
        row = dict(node_ids=ids, sample_id=env_id * num_nodes + ids, environment_index=torch.full((n,), env_id, dtype=torch.long), gold_target=y, gnn_pred=g, llm_pred=l, weight=torch.ones(n), **targets(g, l, y))
        pieces.append(row)
        tabs.append(env['tab70'].float())
        gs.append(env['g'].float())
        ls.append(env['l'].half())
    if not pieces:
        raise ValueError('no observations')
    rows = {k: torch.cat([p[k] for p in pieces]) for k in pieces[0]}
    tab, g, l = (torch.cat(tabs), torch.cat(gs), torch.cat(ls))
    result = dict(dataset=dataset, environment_names=list(ENVIRONMENTS), rows=rows)
    for branch in ('agreement', 'disagreement'):
        mask = rows[branch + '_mask']
        view = tab[mask]
        if branch == 'agreement':
            view = torch.cat((view[:, :27], view[:, 54:]), 1)
        bank = dict(dataset=dataset, split='train', ids=rows['sample_id'][mask], node_ids=rows['node_ids'][mask], environment_index=rows['environment_index'][mask], tab=view, g=g[mask], l=l[mask], gnn_pred=rows['gnn_pred'][mask], llm_pred=rows['llm_pred'][mask], weight=rows['weight'][mask])
        if branch == 'agreement':
            bank['target'] = rows['agreement_target'][mask].bool()
        else:
            for k in ('state', 'trust_target', 'preference_target', 'preference_mask'):
                bank[k] = rows[k][mask]
        result[branch] = bank
    return result
