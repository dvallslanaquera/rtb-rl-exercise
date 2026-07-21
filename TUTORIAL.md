# TUTORIAL.md — RL for ad bidding, explained from zero

A 101 for engineers with no reinforcement-learning background. By the end you'll understand
every design decision in this repo: what a Q-value is, why the network scores *ad features*
instead of having one output per ad, what "offline RL" and the CQL penalty are for, how
cold-start borrowing works, and what happens — step by step — when a bid request arrives.

No RL prerequisites. Comfort with supervised learning ("train a model to predict a label")
is enough.

---

## 1. The problem, in plain terms

Every time someone loads a webpage, an auction happens in ~100 milliseconds. Our side of that
auction receives a **bid request** — *"user U is looking at website W, placement P (say, the
header slot). Want to show an ad? Which one, and what will you pay?"*

We have an inventory of ads. Some fit the page (a NISA investment ad on a finance site), some
fit the user (they read gaming media every day), some are just intrinsically appealing. We
want to pick the ad with the **highest probability of being clicked**, suggest a price, and
answer in single-digit milliseconds.

Two things make this harder than a textbook prediction problem:

1. **We only have logs, not a playground.** We can't freely experiment on live traffic to see
   which ads work — mistakes cost real money. We must learn from *historical* auction logs,
   which were produced by an old, mediocre selection policy.
2. **The market drifts.** New campaigns launch, budgets deplete, competitors change bids. A
   model trained once goes stale, so the system retrains itself every N hours.

## 2. Reinforcement learning vocabulary, mapped to this repo

RL is usually introduced with games: an **agent** observes a **state**, takes an **action**,
receives a **reward**, and learns a **policy** (a rule for choosing actions) that maximizes
reward over time. Here's the translation table — this is 80 % of the jargon you need:

| RL term | Meaning here | Where in code |
|---|---|---|
| **Agent** | The bidding engine | `rl/agent.py` |
| **State** `s` | Everything we know about the request: website embedding, user embedding, user↔site affinity, site CTR stats, market stats, placement | `features/encode.py` |
| **Action** `a` | "Show ad X" — one choice from the candidate set | ad feature vectors |
| **Reward** `r` | `+1` if the impression was clicked, minus a tiny cost term for the price paid | `rl/replay.py` |
| **Policy** `π` | "Score every candidate ad, pick the best" | `serving/inference.py` |
| **Q-value** `Q(s, a)` | *The number the whole system revolves around*: the expected reward if we're in state `s` and take action `a` | `rl/networks.py` |
| **Episode** | A sequence of decisions whose consequences chain together (via campaign budgets) | `sim/env.py` |

The single most important concept: **`Q(s, a)` answers "how good is showing ad `a` right
now?"** If we had a perfect Q function, the optimal policy is trivial — compute Q for every
candidate ad and pick the argmax. Everything in this repo exists to (a) learn a decent Q from
logs, and (b) evaluate it fast.

## 3. From lookup table to neural network

Classic Q-learning maintains a literal table: one cell per (state, action) pair, updated as
experience arrives. That dies immediately here — our state contains continuous embeddings, so
there are effectively infinitely many states, and we'd never see the same one twice.

**Deep Q-Networks (DQN)** replace the table with a neural network: feed in a state, get out
Q-values. The network *generalizes* — it can score a (user, site, ad) combination it has never
seen, because similar users and similar ads get similar embeddings and therefore similar
Q-values. Training looks like supervised regression: for each logged impression we know the
state, the action taken, and the reward observed, so we nudge `Q(s, a_logged)` toward what
actually happened.

### One important departure from the textbook DQN

The classic DQN (the Atari one) has a **fixed output head**: one output neuron per possible
action. That assumes the action set never changes. Our "actions" are ads — and ad inventory
changes daily. A new campaign would mean a new output neuron, i.e. a new architecture and a
retrain from scratch. Cold-start would be architecturally impossible.

So instead, this repo's network scores a **(state, ad-features) pair**:

```
Q(state, ad_features) → one scalar
```

An ad is represented by what it *is* (its content embedding, its category-match scores, its
smoothed historical CTR, its bid cap...) rather than by *which output slot it occupies*. To
choose among 40 candidates, we run one batched forward pass over 40 (state, ad) pairs and take
the argmax. Consequences:

- **Inventory can grow or shrink freely** — new ad, same network.
- **Cold-start becomes a feature problem, not an architecture problem** (see §7).
- This shape of model is sometimes called a *scoring* or *pointwise ranking* network — if
  you've built search rankers, it should feel familiar. The RL part is *what the target
  label means* (expected reward) and *how we correct for learning from biased logs* (§5).

### The per-ad id-embedding

Content features can't explain everything — two ads in the same category with similar creative
text can perform very differently (brand strength, creative quality, price point). The network
therefore also learns a small **id-embedding** per known ad (a 16-dim trainable vector, like a
word embedding), concatenated to the content features. Think of it as a learned "residual
appeal" score with capacity to absorb what content can't. It's also the quantity cold-start
borrows for brand-new ads (§7).

### Dueling architecture (a refinement, not a pillar)

The network splits its estimate into `V(s)` — "how promising is this request overall?" — and
`A(s, a)` — "how much better/worse is this specific ad than average here?", combined as
`Q = V + A − mean(A)`. Intuition: some requests are just good (engaged user on a high-quality
site — everything clicks) and some are bad; separating that shared component from the per-ad
difference makes learning more sample-efficient. If you drop this (`dueling: false` in config),
everything still works, slightly worse.

## 4. What the reward is — and why only "won" impressions

For each logged impression the training reward is:

```
r = 1.0 · click − 0.00005 · price_paid_jpy
```

Click is the objective; the tiny cost term is a regularizer that breaks ties toward cheaper
impressions rather than a serious economic model.

Training uses **only auctions we won**. Why: a click is *observable only if the ad was actually
shown*. For lost auctions there is no label — we'll never know whether the user would have
clicked. So `Q(s, a)` learns "expected click value *given the ad is served*", which is exactly
the ranking signal the bidder needs. (Honest footnote: which impressions we won correlates with
price and placement, so this conditional distribution is mildly skewed — a known,
uncorrected-for bias, listed in [ARCHITECTURE.md](ARCHITECTURE.md) §7.)

## 5. The heart of the matter: learning from logs (offline RL) and CQL

This is the part where RL genuinely differs from supervised learning, and it's the most
important section of this tutorial.

### The problem: you can't trust Q-values for actions the logs never took

Our logs were produced by an old behavior policy. Suppose that policy almost never showed
gaming ads on finance sites. Then the training data contains almost no (finance-site,
gaming-ad) pairs — and for those pairs the network's Q output is **unconstrained
extrapolation**. Neural networks don't say "I don't know"; they say *something*, and it's often
something wildly optimistic.

Now recall how our policy works: **argmax over candidates**. Argmax is a machine for finding
the largest value — including the largest *error*. If one out-of-distribution ad gets a
hallucinated Q of 3.7, the policy will confidently pick it, every time. In online RL the agent
would try it, get no clicks, and correct itself. **Offline, there is no correction** — we ship
the hallucination to production. This failure mode (value overestimation on out-of-distribution
actions) is *the* central problem of offline RL.

### The fix: Conservative Q-Learning (CQL)

CQL adds one term to the loss. For each training example, alongside the normal regression term,
we compute Q for the logged ad *and* for a pool of other candidate ads, and penalize:

```
loss = MSE(Q(s, a_logged), r)                      ← "fit what actually happened"
     + α · [ logsumexp_over_pool(Q(s, ·)) − Q(s, a_logged) ]   ← "be humble about the rest"
```

`logsumexp` is a soft maximum — it's large when *any* ad in the pool has a large Q. So the
penalty says: **push down whatever currently looks best among arbitrary ads, and push up the
ad we have actual evidence for.** Ads with real logged support get their values restored by the
regression term; ads without support just get pushed down. The equilibrium: Q stays accurate
where data exists and pessimistic where it doesn't — which is exactly the prior you want when
an argmax is hunting for outliers. `α` (config `rl.cql_alpha`, default 1.0) sets how paranoid
to be.

Trade-off worth knowing: CQL deliberately *biases Q downward*, so the outputs are no longer
calibrated click probabilities — they're conservative ranking scores. Fine for choosing an ad;
misleading if you read them as probabilities (see §8).

### Where the candidate pool comes from during training

Each minibatch row puts the logged ad in column 0 and fills the remaining columns with
randomly sampled known ads ([`rl/replay.py`](src/rtb_rl/rl/replay.py)). Column 0's Q feeds the
regression term; the whole row feeds the logsumexp. That's the entire CQL implementation —
about three lines in [`rl/agent.py`](src/rtb_rl/rl/agent.py).

## 6. The sequential story: budgets, γ, and Double-DQN (and honesty about it)

Everything so far treats each impression as independent: predict reward, pick argmax. That's
technically a **contextual bandit** — RL's one-step special case, and the right mental model
for what this repo's training loop *actually does today*.

Full RL enters when decisions **now** change what's possible **later**. The real coupling in
ad delivery is **campaign budgets**: serving an expensive ad this second depletes budget that
constrains tonight's impressions. The machinery for that is present in the codebase:

- The TD target generalizes from `y = r` to `y = r + γ · max_a' Q(s', a')` — "reward now plus
  discounted value of the best next decision". γ (`rl.gamma`, 0.85) is how much the future
  matters.
- Naively using the same network to both *choose* `a'` and *evaluate* it inflates values
  (argmax again picks the biggest error — same disease as §5, different limb). **Double-DQN**
  splits the roles: the online network selects `a'`, a slowly-updated **target network** copy
  evaluates it. This is the standard, cheap fix.
- [`sim/env.py`](src/rtb_rl/sim/env.py) is a Gymnasium environment where budgets actually
  deplete step by step — the sequential MDP these pieces are built for.

**Honest boundary** (same one documented in [ARCHITECTURE.md](ARCHITECTURE.md) §7): the current
training pipeline feeds only one-step transitions from logs, so γ, the target network, and the
env are dormant — wired, unit-tested, but not driving production training. If someone tells
you "this repo is a DQN", the precise answer is: *it's a CQL-regularized neural contextual
bandit today, with the DQN machinery in place for when budget pacing enters the training loop.*

## 7. Cold start: scoring things that have no history

**New ad** (launched an hour ago, zero impressions): it has no learned id-embedding and no CTR
history. Solution — *borrow from lookalikes*: embed its creative text, find the K=5 most
similar known ads by content cosine similarity, and give the new ad the similarity-weighted
average of their id-embeddings and CTR priors ([`rl/cold_start.py`](src/rtb_rl/rl/cold_start.py)).
Because the network scores feature vectors (§3), we can just *feed it this synthetic
id-embedding* — no retraining, scored sensibly on its first impression, marked with an
`is_cold` flag. As real history accrues, empirical-Bayes smoothing gradually replaces the
borrowed prior with actual data.

**New user** (no engagement history → zero vector): assume they're aligned with what they're
reading *right now* — use the website's embedding as a stand-in user vector. A deliberately
simple prior that decays automatically as engagement accrues.

## 8. Serving: from Q-values to a bid

At request time ([`serving/inference.py`](src/rtb_rl/serving/inference.py)):

1. Assemble the state from the warm in-memory feature snapshot (no database on the hot path).
2. Build the candidate matrix — every eligible ad's features + id-embedding, cold ads via §7.
3. One batched forward pass → Q per candidate → **argmax picks the ad**.
4. Suggest a price: `clip(floor + bid_cap · clip(Q, 0, 1), floor, bid_cap)` — bid more
   aggressively the more valuable the impression looks, never below the floor or above the
   campaign's cap. (A deliberately simple heuristic: real bidders price as
   `p(click) × value_per_click`, which requires the calibration CQL traded away — §5.)

Latency budget: the model and features live in process memory, so the whole path is dict
lookups + one matmul — sub-millisecond on CPU for ~40 candidates.

## 9. Retraining: tracking a moving market

Every N hours ([`pipelines/retrain_loop.py`](src/rtb_rl/pipelines/retrain_loop.py)):

```
rebuild features → warm-start train from current model → evaluate in simulator
      → promote only if it clears the uplift gate → serving hot-swaps, no restart
```

- **Warm start**: initialize from the incumbent's weights so each cycle adapts incrementally.
- **Sim gate**: a candidate must demonstrate CTR uplift in the offline simulator before the
  registry pointer flips. (Current gate compares against a random baseline, not the incumbent —
  a known gap, [ARCHITECTURE.md](ARCHITECTURE.md) §7.)
- **Hot swap**: serving polls the registry pointer and reloads the model in the background.

## 10. One request, end to end

A concrete trace, using the demo's synthetic world:

> Request: user `u000123` (reads finance and tech media), website `w0007`
> (`finance007.example.jp`), placement `header`.

1. **State**: `w0007`'s content embedding ‖ `u000123`'s engagement-weighted embedding ‖
   their cosine affinity (high — a finance reader on a finance site) ‖ the site's base
   CTR/CVR ‖ market win-rate and clearing price for `(w0007, header)` ‖ placement one-hot.
2. **Candidates**: 40 eligible ads, including `ad117` — a brand-new NISA campaign with zero
   history, scored via its 5 nearest known finance ads' borrowed id-embeddings.
3. **Forward pass**: Q for all 40 pairs in one matmul. The finance ads score high (content
   match × user affinity), `ad117` lands near its established lookalikes, a footer-optimized
   discount ad scores low.
4. **Decision**: argmax → say, `ad031` (established finance ad, strong id-embedding), Q = 0.62,
   bid cap ¥180 → suggested bid ≈ ¥141. Response returned in ~1 ms with the model version and
   a `cold_start` flag.
5. **Later**: tonight's retrain cycle folds new logs into features, warm-starts training, and —
   if the simulator gate passes — tomorrow's requests are scored by the updated policy.

## 11. Glossary

| Term | One-liner |
|---|---|
| **Q-value** | Expected reward for taking a specific action in a specific state. |
| **Policy** | The rule mapping states to actions; here, argmax over candidate Q-values. |
| **Contextual bandit** | One-step RL: pick an action given context, observe reward, no consequences carried forward. What this repo's training loop is today. |
| **MDP** | Markov Decision Process — the sequential setting where actions affect future states (here: budgets). |
| **Offline (batch) RL** | Learning a policy purely from logged data, without live interaction. |
| **OOD actions** | Actions (ads) the logging policy rarely/never took — where Q estimates are extrapolation and can't be trusted. |
| **CQL** | Conservative Q-Learning: a loss penalty keeping Q pessimistic on unsupported actions so the argmax can't chase hallucinations. |
| **Target network** | A slow-moving copy of the Q-network used to compute bootstrap targets, stabilizing sequential training. |
| **Double-DQN** | Select the next action with the online net, evaluate it with the target net — reduces overestimation. |
| **Dueling** | Decompose Q into state value + per-action advantage. |
| **γ (gamma)** | Discount factor: how much future reward counts vs immediate. γ=0 ⇒ bandit. |
| **Warm start** | Initializing training from the previous model's weights. |
| **Behavior policy** | The (old) policy that produced the training logs. |
| **Off-policy evaluation / SNIPS** | Estimating how a *new* policy would have performed using logs produced by an *old* one, via importance weighting. |

## 12. Where to read next

- [ARCHITECTURE.md](ARCHITECTURE.md) — full component map, serving hot path, known gaps
  (§7 there is the honest list), and the production GCP/GKE topology.
- [README.md](README.md) — quickstart, commands, results table.
- Suggested code-reading order, matching this tutorial's arc:
  [`features/encode.py`](src/rtb_rl/features/encode.py) (what the model sees) →
  [`rl/networks.py`](src/rtb_rl/rl/networks.py) (§3) →
  [`rl/replay.py`](src/rtb_rl/rl/replay.py) (§4–5) →
  [`rl/agent.py`](src/rtb_rl/rl/agent.py) (§5–6) →
  [`rl/cold_start.py`](src/rtb_rl/rl/cold_start.py) (§7) →
  [`serving/inference.py`](src/rtb_rl/serving/inference.py) (§8) →
  [`pipelines/retrain_loop.py`](src/rtb_rl/pipelines/retrain_loop.py) (§9).
