None selected 

Skip to content
Using Brown University Mail with screen readers
enyan_zhang1@brown.edu 

1 of 39
manifold muon online
Inbox

Prakash, Arjun <arjun_prakash@brown.edu>
Mon, May 18, 8:58 PM (7 days ago)
to Luke

Hey Luke, 

Can you send me a skills.md file for a standalone online manifold muon implementation. I think I need to use it for the pixel stuff. 

I am to have a branch set up for you tomorrow/wednesday. 

Cheers,
A

Zhang, Luke
Attachments
Mon, May 18, 9:38 PM (7 days ago)
to me

Hi Arjun, 

Attached is the skill. I've defaulted the skill to use no LR transfer scale factor for manifold muon; you can explicitly ask to change it if you want. 

Best,
Luke
 One attachment
  •  Scanned by Gmail



# Standalone Manifold Muon Online

## Intent and Source

Source: Jeremy Bernstein, ["Modular Manifolds"](https://thinkingmachines.ai/blog/modular-manifolds/),
Thinking Machines Lab, Sep 26, 2025, DOI `10.64434/tml.20250926`.

The algorithm is meant to make linear-layer training scale predictable. A linear
weight acts as a vector multiplier, so the post chooses the Stiefel manifold to
keep its singular values at one, and uses the spectral norm to measure how large
an update can be as an operator. The resulting update should be the tangent
matrix that most decreases the loss while satisfying both constraints:

- the update is tangent to the Stiefel constraint, `A.T @ W + W.T @ A = 0`;
- the update has bounded spectral norm, implemented through `matrix_sign`;
- the post-step weight is retracted back to the manifold with `matrix_sign(W)`.

Dual ascent is the mechanism for finding the Lagrange multiplier `Lambda` that
enforces the tangent constraint. The online implementation keeps `Lambda` and a
dual velocity `V` as optimizer state and performs one momentum-style dual update
per training step, instead of solving the dual problem to convergence every
minibatch. This follows the post's practical motivation: dual ascent adds
per-step overhead, and that overhead can be reduced by adding momentum and
running dual ascent online.

## Core Objects

For one linear weight matrix `W` with shape `[fanout, fanin]` and raw gradient
`G` of the same shape, keep optimizer state for both the primal gradient
momentum and the online dual solve:

- `Lambda`: symmetric dual variable with shape `[k, k]`.
- `V`: symmetric dual velocity with shape `[k, k]`.
- `M`: solver-gradient momentum with the same shape as `W`.
- `k = min(fanout, fanin)` after orienting `W` into tall form.

For the online method, initialize `Lambda`, `V`, and `M` to zeros. `Lambda` and
`V` live in the oriented coordinate system; `M` lives in the original weight
orientation. Do not use the offline initializer
`-0.25 * (W.T @ G + G.T @ W)` for the online state; that initializer belongs
to the finite-iteration `dual_ascent_tangent` and ADMM variants.

## Orientation

Always run the dual step on a tall matrix with rows greater than or equal to
columns:

```python
def orient_tall(matrix):
    transposed = matrix.shape[-2] < matrix.shape[-1]
    return (matrix.T if transposed else matrix), transposed


def restore_orientation(matrix, transposed):
    return matrix.T if transposed else matrix
```

Apply the same orientation to `W` and the gradient-like matrix passed to the
dual step, which is the momentum-smoothed `M` in the full update. The returned
tangent is restored to the original orientation, but the state stays in the
oriented `[k, k]` coordinates.

## Matrix Sign

`msign` is the matrix sign / polar factor, not an elementwise sign. Use the
Polar Express polynomial iteration below. Keep the base coefficients and
stabilization formula exactly as written:

```python
_ABC_LIST = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

ABC_LIST_STABLE = [
    (float(a) / 1.01, float(b) / 1.01**3, float(c) / 1.01**5)
    if idx < len(_ABC_LIST) - 1
    else (float(a), float(b), float(c))
    for idx, (a, b, c) in enumerate(_ABC_LIST)
]
```

This routine orients the input to the left-polynomial form, normalizes by
`frobenius_norm * 1.01`, runs 10 steps by default, casts back to the input dtype,
and applies `nan_to_num`:

```python
def matrix_sign(M, steps=10):
    transposed = M.shape[-2] > M.shape[-1]
    X = M.T if transposed else M
    norm = frobenius_norm(X)
    norm = 1.0 if norm == 0 else norm
    X = X / (norm * 1.01)
    I = eye(X.shape[-2], dtype=X.dtype)

    for step in range(steps):
        a, b, c = ABC_LIST_STABLE[min(step, len(ABC_LIST_STABLE) - 1)]
        S = X @ X.T
        Y = (c * S + b * I) @ S + a * I
        X = Y @ X

    X = X.T if transposed else X
    return nan_to_num(X.astype(M.dtype))
```

Use library equivalents for `frobenius_norm`, `eye`, and `nan_to_num` when
porting to PyTorch or NumPy.

## Online Dual Step

The online dual step is:

```python
def sym(A):
    return 0.5 * (A + A.T)


def online_dual_ascent_step_tall(W, G, Lambda, V, alpha, beta):
    Lambda = sym(Lambda)
    V = sym(V)

    Lambda_tilde = sym(Lambda + beta * V)
    tangent = matrix_sign(G + 2.0 * W @ Lambda_tilde)

    H = sym(W.T @ tangent + tangent.T @ W)
    V_next = sym(beta * V - alpha * H)
    Lambda_next = sym(Lambda + V_next)
    return tangent, Lambda_next, V_next
```

The sign of the residual update is important: `V_next = beta * V - alpha * H`.
Changing this to `+ alpha * H` moves the dual variable in the wrong direction.

A complete orientation-safe step is:

```python
def online_dual_ascent_step(W, G, state, alpha=1e-2, beta=0.9):
    W_tall, transposed = orient_tall(W)
    G_tall = G.T if transposed else G
    Lambda, V = state
    tangent_tall, Lambda_next, V_next = online_dual_ascent_step_tall(
        W_tall, G_tall, Lambda, V, alpha, beta
    )
    return restore_orientation(tangent_tall, transposed), (Lambda_next, V_next)
```

## Full Online Manifold Muon Update

Use the momentum and decoupled weight-decay version as the standalone update.
Here `array`, `zeros`, `sqrt`, and `zeros_like` refer to the chosen tensor
backend:

```python
def init_manifold_muon_online_state(W):
    W_tall, _ = orient_tall(W)
    k = W_tall.shape[-1]
    Lambda = zeros((k, k), dtype=W.dtype)
    V = zeros_like(Lambda)
    M = zeros_like(W)
    return {"dual": (Lambda, V), "momentum": M}


def scale_radius(W, scale):
    if scale == "none":
        return array(1.0, dtype=W.dtype)
    if scale == "ratio":
        return sqrt(W.shape[0] / W.shape[1])
    raise ValueError(f"unknown manifold Muon scale: {scale}")


def retract_to_scaled_stiefel(W, scale):
    radius = scale_radius(W, scale)
    return radius * matrix_sign(W)


def manifold_muon_online_update(
    W,
    G,
    state,
    *,
    lr,
    alpha=1e-2,
    beta=0.9,
    momentum=0.9,
    weight_decay=0.01,
    scale="none",
):
    alpha = array(alpha, dtype=W.dtype)
    beta = array(beta, dtype=W.dtype)
    momentum = array(momentum, dtype=W.dtype)
    radius = scale_radius(W, scale)

    Lambda, V = state["dual"]
    M = momentum * state["momentum"] + G
    unit_W = W / radius
    tangent, dual_next = online_dual_ascent_step(unit_W, M, (Lambda, V), alpha, beta)

    update = radius * tangent + weight_decay * W
    W_next = W - lr * update
    W_next = retract_to_scaled_stiefel(W_next, scale)
    state_next = {"dual": dual_next, "momentum": M}
    return W_next, state_next
```

The update order is fixed: form momentum-smoothed gradient `M`, compute the
tangent from `M`, add decoupled weight decay to the primal update, step `W`,
then retract with `retract_to_scaled_stiefel(W_next, scale)`.

## Scale Choice and Radius

The `scale` option selects the manifold radius and the tangent multiplier.
Default to `scale="none"`:

- `scale="none"`: use unit Stiefel weights, so `radius = 1`; retract with
  `matrix_sign(W_next)`.
- `scale="ratio"`: use RMS-radius weights, so
  `radius = sqrt(fanout / fanin)`; solve the tangent problem against
  `unit_W = W / radius`, multiply the tangent update by `radius`, and retract
  with `radius * matrix_sign(W_next)`.

Initialize or project stored weights with the same rule:
`W = retract_to_scaled_stiefel(raw_W, scale)`. This means `scale="none"` stores
unit-Stiefel weights, while `scale="ratio"` stores RMS-radius weights.

Do not include other scale modes unless the caller explicitly asks for an
extension. When reporting an implementation to a user, state which scale mode
was used because it changes the stored weight radius.

## Defaults and Verification

- Dual-step defaults: `alpha=1e-2`, `beta=0.9`.
- Standalone update defaults: `momentum=0.9`, `weight_decay=0.01`,
  `scale="none"`.
- Smoke-test setting: repeated online steps with `alpha=5e-2`, `beta=0.0`
  should not increase the tangent constraint residual.

Check these invariants in a standalone implementation:

- `Lambda` and `V` stay symmetric after every step.
- The tangent has the same shape and dtype as `W`.
- `unit_W.T @ tangent + tangent.T @ unit_W` is finite, and with fixed `W, G`,
  repeated online steps using `alpha=5e-2`, `beta=0.0` do not increase the
  final residual relative to the first online residual.
- After retraction, `W / radius` is on the unit Stiefel constraint in the
  current orientation.
- Wide and tall matrices both use the same state shape `[min(fanout, fanin),
  min(fanout, fanin)]`.

## Common Mismatches

- Do not treat `matrix_sign` as `sign(x)` elementwise.
- Do not initialize online `Lambda` from the offline dual initializer unless you
  deliberately want a custom warm start.
- Do not store a transposed state for wide matrices; store the state in the
  oriented coordinates and only transpose tangents back.
- Do not omit the final retraction if the optimizer should keep `W` on the
  manifold.
- Do not apply the ratio scale twice. For `scale="ratio"`, the radius already
  appears in `unit_W = W / radius`, `update = radius * tangent + ...`, and
  `W_next = radius * matrix_sign(W_next)`.
skills.md
Displaying skills.md.