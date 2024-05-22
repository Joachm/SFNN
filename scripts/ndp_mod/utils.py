from typing import TypeAlias
from jaxtyping import PyTree
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
from src.gnn.utils import to_networkx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

Graph: TypeAlias=PyTree

def draw_with_embedding_as_pos(G: Graph, show: bool=True, reducer_factory=PCA, **kwargs):
	nxG = to_networkx(G)
	if G.h.shape[-1] > 2:
		reducer = reducer_factory(n_components=2)
		pos = reducer.fit_transform(np.array(G.h))
	elif G.h.shape[-1]==2:
		pos = G.h
	else:
		raise ValueError
	nx.draw(nxG, pos=pos, **kwargs)
	if show: plt.show()

def draw_mlp(G: Graph, I: int, O: int, L: int, show: bool=True, seed: int=1, **kwargs):
	N = G.N
	H = (N - I - O)//L
	nxG = to_networkx(G)
	p = [(-0.5, i/(I-1)-0.5) for i in range(I)]
	xs = jnp.linspace(-.2, .2, L) if L>1 else [0.]
	for l in range(L):
		hp = [(float(xs[l]), 2*(h/(H-1)-0.5)) for h in range(H)]
		p = p + hp
	if O > 1:
		p = p + [(0.5, i/(O-1)-0.5) for i in range(O)]
	else:
		p = p + [(0.5, 0.)]
	nx.draw(nxG, pos=p, **kwargs)
	if show:
		plt.show()


def draw_reservoir(G, I: int, O: int, R: int=1, show: bool=True, use_spring: bool=False, **kwargs):
    N = G.N
    H = (N - I - O)//R
    nxG = to_networkx(G)
    p = [(-0.5, i/(I-1)-0.5) for i in range(I)]
    if R==1:
        if not use_spring:
            hp = jr.uniform(jr.key(1), (H, 2), minval=jnp.array([-0.3, -1.]), maxval=jnp.array([0.3, 1.]))
            p = p + [(float(hp[i,0]), float(hp[i,1])) for i in range(H)]
        else:
            hp = [(-0.3, 0.) if any([bool(G.A[i,j]) for i in range(I)]) else (0.3, 0.) if any([bool(G.A[j,i]) for i in range(G.N-O, G.N)]) else (0.,0.) for j in range(I, H+I)]
            p = p+hp
    else:
        raise NotImplementedError
    p = p + [(0.5, i/(O-1)-0.5) for i in range(O)]
    if use_spring:
        p = nx.spring_layout(nx.to_undirected(nxG), pos={i:p[i] for i in range(N)}, 
                             fixed=list(range(I))+list(range(I+H*R, I+H*R+O)),
                             scale=1, k=.2)
    nx.draw(nxG, pos=p, **kwargs)
    if show:
        plt.show()