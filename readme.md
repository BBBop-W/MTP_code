# Formulation

$$
\begin{align}
\max &\sum\limits_{i\in \mathcal I}\sum\limits_{j \in \mathcal J}\sum\limits_{h\in\mathcal H}x_{ij}^hl_i\\
\text{s.t.  }   &\sum\limits_{j\in \mathcal J}\sum\limits_{h\in\mathcal H}x_{ij}^h \leq C_i+D_i, &&\forall i \in \mathcal I, \\
&\sum\limits_{j \in \mathcal J}\sum\limits_{h\in\mathcal H}x_{ij}^h\geq D_i, &&\forall i \in \mathcal I,\\
&\sum\limits_{i \in \mathcal I}\sum\limits_{h\in\hat{\mathcal H}}x_{ij}^h(l_i + \Delta)\leq \sum\limits_{h\in\hat{\mathcal H}}L_h - \Delta, &&\forall j \in \mathcal J, \hat{\mathcal H}\in\{\mathcal H_1, {\mathcal H_3\}}, \\
&\sum\limits_{i\in \mathcal I}\sum\limits_{h\in { \mathcal H_4^k}}x_{ij}^h(l_i + \Delta)\leq \sum\limits_{h\in {\mathcal H_4^k}}L_h ,&&\forall j \in \mathcal J, k \in \mathcal K,  \\
&\sum\limits_{i \in \mathcal I}\sum\limits_{h\in\hat{\mathcal H}}x_{ij}^h(l_i + \Delta)\leq
\sum\limits_{h\in\hat{\mathcal H}}L_h + \Delta , &&\forall j \in \mathcal J,k \in \mathcal K,\hat{\mathcal H}\in\{ {\mathcal H_5^k},{\mathcal H_2, } \{\mathbf C\}\},\\
&\sum\limits_{i\in \mathcal I}\sum\limits_{h\in {\mathcal H_6^k}}x_{ij}^h(l_i + \Delta)\leq \sum\limits_{h\in { \mathcal H_6^k}}L_h +{M}(1 - \pi_{jk}), &&\forall j\in \mathcal J,k \in \mathcal K,\\
&\sum\limits_{i \in \mathcal I}x_{ij}^{\mathbf E}(l_i + \Delta)\leq L_{\mathbf E} + \Delta + {M}(1 - \pi_{jk}), &&\forall j \in \mathcal J,\\
&x_{ij}^h\leq N\epsilon_i^h+{N} \pi_{jk}, &&\forall i\in \mathcal I,j \in \mathcal J,k \in \mathcal K,h \in \mathcal H^k,\\
&x_{ij}^h\leq N\phi_i^h+{N}(1-\pi_{jk}), &&\forall i \in \mathcal I,j \in \mathcal J,k \in \mathcal K,h\in \mathcal H^k,\\
&\pi_{jk} \in \{0,1\},  &&\forall j\in \mathcal J, k \in \mathcal K,\\
&x_{ij}^h \in \mathbb N_+,  &&\forall i\in \mathcal I,j \in \mathcal J,h \in \mathcal H.
\end{align}
$$
