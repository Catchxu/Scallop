import torch
import numpy as np
import pandas as pd


class GeneTokenizer:
    """
    Gene tokenizer: Convert gene names into token indices.
    Output shape: (C, L)
    """
    def __init__(self, token_dict: pd.DataFrame, pad_token="<pad>", max_length=4096):
        """
        token_dict: DataFrame with columns: token_index, gene_symbol, gene_id
        """
        self.max_length = max_length

        # pad token index
        pad_row = token_dict[token_dict["gene_id"] == pad_token]
        if len(pad_row) == 0:
            raise ValueError("pad token not found in token_dict")
        self.pad_index = int(pad_row["token_index"].iloc[0])

        # build two lookup tables for fast search
        self.symbol2id = dict(zip(token_dict["gene_symbol"], token_dict["token_index"]))
        self.id2id = dict(zip(token_dict["gene_id"], token_dict["token_index"]))

    def __call__(self, adata, gene_key=None, order_matrix=None):
        """
        adata: AnnData, shape (C, G)
        gene_key: if not None, fetch gene names from adata.var[gene_key]
        order_matrix: CxG matrix giving order per cell. If provided,
                      rearrange gene names in each row accordingly.
        """
        # ---- 1. Get gene names ----
        if gene_key is None:
            gene_names = adata.var_names.tolist()
        else:
            gene_names = adata.var[gene_key].tolist()

        G = len(gene_names)
        if G > self.max_length:
            raise ValueError(f"G={G} exceeds max_length={self.max_length}")

        # ---- 2. Detect gene name type: symbol or ENSG id ----
        # If all names start with 'ENSG', treat as gene_id
        use_gene_id = all(name.startswith("ENSG") for name in gene_names)

        # Select lookup table
        lookup = self.id2id if use_gene_id else self.symbol2id

        # ---- 3. Resolve token index for each gene ----
        # Unknown genes are mapped to pad
        base_order_idx = np.array([lookup.get(g, self.pad_index) for g in gene_names])  # shape G

        C = adata.n_obs
        tokens = np.full((C, self.max_length), self.pad_index, dtype=np.int64)

        # ---- 4. Apply per-cell ordering if provided ----
        # order_matrix is assumed to be C×G, containing column indices
        if order_matrix is not None:
            assert order_matrix.shape == (C, G)
            for i in range(C):
                ordered_indices = order_matrix[i]  # index of genes
                row = base_order_idx[ordered_indices]
                tokens[i, :G] = row
        else:
            # no special ordering, broadcast the same sequence to all cells
            tokens[:, :G] = base_order_idx

        # ---- 5. Pad mask: 1 means padded ----
        pad_mask = np.zeros((C, self.max_length), dtype=np.bool_)
        pad_mask[:, G:] = True

        return torch.tensor(tokens), torch.tensor(pad_mask)


class ExprTokenizer:
    """
    Expression tokenizer: convert adata.X (CxG) into padded matrix.
    Output shape: (C, L), pad filled with 0.
    """
    def __init__(self, max_length=4096):
        self.max_length = max_length

    def __call__(self, adata, order_matrix=None):
        X = adata.X  # C×G
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        C, G = X.shape
        if G > self.max_length:
            raise ValueError(f"G={G} exceeds max_length={self.max_length}")

        # ---- 1. Per-cell ordering if provided ----
        if order_matrix is not None:
            assert order_matrix.shape == (C, G)
            X_ordered = np.zeros_like(X)
            for i in range(C):
                X_ordered[i] = X[i, order_matrix[i]]
            X = X_ordered

        # ---- 2. Pad to max_length ----
        expr = np.zeros((C, self.max_length), dtype=np.float32)
        expr[:, :G] = X

        pad_mask = np.zeros((C, self.max_length), dtype=np.bool_)
        pad_mask[:, G:] = True

        return torch.tensor(expr), torch.tensor(pad_mask)




class CondTokenizer:
    """
    Condition tokenizer:
    Inputs: adata + obs keys: platform_key, species_key, tissue_key, disease_key
    Output: Cx4 tensor (4 condition tokens per cell)
    """
    def __init__(self, cond_dict=None, simplify=False):
        """
        cond_dict: DataFrame with columns: cond_value, token_index
        simplify: if True → always return 0 token for all 4 features
        """
        self.simplify = simplify

        # If no dict provided, create an empty one with a reserved 0 token
        if cond_dict is None:
            cond_dict = pd.DataFrame(
                {"cond_value": ["<unk>"], "token_index": [0]}
            )

        # Ensure pad token exists
        if "<unk>" not in cond_dict["cond_value"].values:
            raise ValueError("cond_dict must contain '<unk>' as token_index=0")

        self.cond_dict = cond_dict

    def _get_next_index(self):
        """Return next available token index."""
        return int(self.cond_dict["token_index"].max()) + 1

    def _fetch_or_add(self, value):
        """
        Lowercase the value, check exist.
        If not exist, add new token row.
        """
        value = str(value).lower()

        # missing or nan → return 0 token
        if value == "nan":
            return 0

        df = self.cond_dict
        hit = df[df["cond_value"] == value]

        if len(hit) > 0:
            return int(hit["token_index"].iloc[0])

        # add new token
        new_idx = self._get_next_index()
        new_row = pd.DataFrame({"cond_value": [value], "token_index": [new_idx]})
        self.cond_dict = pd.concat([self.cond_dict, new_row], ignore_index=True)
        return new_idx

    def __call__(
            self, 
            adata, 
            platform_key=None, 
            species_key=None,
            tissue_key=None, 
            disease_key=None
    ):
        """
        Return: Cx4 tensor
        """
        C = adata.n_obs

        # If simplify mode: return all zero tokens
        if self.simplify:
            return torch.zeros((C, 4), dtype=torch.long)

        obs = adata.obs

        keys = [platform_key, species_key, tissue_key, disease_key]
        cond_values = []

        for key in keys:
            if key is None or key not in obs:
                # missing key → use pad (0)
                cond_values.append(["nan"] * C)
            else:
                cond_values.append(obs[key].astype(str).tolist())

        # cond_values is list of 4 lists, each length C
        out = np.zeros((C, 4), dtype=np.int64)

        # For each condition type
        for j in range(4):
            for i in range(C):
                out[i, j] = self._fetch_or_add(cond_values[j][i])

        return torch.tensor(out, dtype=torch.long)


class BatchTokenizer:
    """
    Assign a unique batch ID (integer counter) per adata input.
    Output: Cx1 tensor filled with batch index.
    """
    def __init__(self, simplify=False):
        self.counter = 0
        self.simplify = simplify

    def __call__(self, adata):
        C = adata.n_obs

        # In simplify mode: always output zero token
        if self.simplify:
            return torch.zeros((C, 1), dtype=torch.long)

        # Normal mode: assign increasing batch_id
        batch_id = self.counter
        self.counter += 1

        out = np.full((C, 1), batch_id, dtype=np.int64)
        return torch.tensor(out)




if __name__ == "__main__":
    import scanpy as sc
    import pandas as pd
    import torch

    print("Loading data...")
    adata = sc.read_h5ad("/data1021/xukaichen/data/DRP/cell_line.h5ad")
    token_dict = pd.read_csv("./resources/token_dict.csv")

    gene_tokenizer = GeneTokenizer(token_dict)
    expr_tokenizer = ExprTokenizer()

    print("Tokenizing genes...")
    gene_tokens, gene_pad = gene_tokenizer(adata)

    print("Tokenizing expression...")
    expr_tokens, expr_pad = expr_tokenizer(adata)

    # ============================================================
    # =============== Basic Shape Checks ==========================
    # ============================================================
    C, G = adata.X.shape
    L = gene_tokens.shape[1]

    print(f"Cells: {C}, Genes: {G}, Max Length: {L}")

    assert gene_tokens.shape == (C, L), "gene_tokens shape mismatch"
    assert expr_tokens.shape == (C, L), "expr_tokens shape mismatch"
    assert gene_pad.shape == (C, L), "gene_pad shape mismatch"
    assert expr_pad.shape == (C, L), "expr_pad shape mismatch"

    print("✔ Shape checks passed")

    # ============================================================
    # =============== Pad Mask Consistency Check =================
    # ============================================================
    # GeneTokenizer uses True for padding
    # ExprTokenizer also uses True
    # Therefore both pad masks must be identical
    # ============================================================

    same_pad = torch.equal(gene_pad, expr_pad)

    if same_pad:
        print("✔ gene_pad and expr_pad are IDENTICAL")
    else:
        print("❌ gene_pad and expr_pad differ!")
        # locate difference
        diff = (gene_pad != expr_pad)
        idx = torch.nonzero(diff)
        print(f"   Difference found at positions: {idx[:10]}")
        raise ValueError("pad mask mismatch")

    # ============================================================
    # =============== Gene Padding Sanity Check ==================
    # ============================================================
    # gene_pad[:, :G] must be False
    # gene_pad[:, G:] must be True
    # ============================================================

    assert not gene_pad[:, :G].any(), "Front G region should NOT be padded for genes"
    assert gene_pad[:, G:].all(), "Back region must be padded for genes"

    print("✔ Gene padding mask sanity checks passed")

    # ============================================================
    # =============== Expression Padding Sanity Check ============
    # ============================================================
    # expr_pad[:, :G] must be False
    # expr_pad[:, G:] must be True
    # Zero padding check for expr_tokens
    # ============================================================

    assert not expr_pad[:, :G].any(), "Front G region should NOT be padded for expr"
    assert expr_pad[:, G:].all(), "Back region must be padded for expr"

    # Check zeros in padded expr region
    assert torch.all(expr_tokens[:, G:] == 0), "padded expression must be 0"

    print("✔ Expression padding and zero-fill checks passed")

    # ============================================================
    # =============== Gene and Expression pad alignment ==========
    # ============================================================
    print("FINAL CHECK: pad masks fully aligned ✔\nAll tests passed!\n")
    