def convert_to_xgi(hoi, model, minsize=3, maxsize=5, n_best=10):
    from hoi.utils import get_nbest_mult
    import xgi
    
    df = get_nbest_mult(hoi, model=model, minsize=minsize, maxsize=maxsize, n_best=n_best)
    H = xgi.Hypergraph()
    for i, row in df.iterrows():
        H.add_edge(row['multiplet'])
    return H





