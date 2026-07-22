# TUTORIAL_SIMPLE.md — How this app works, in plain English

This is the short, no-jargon version. It explains what the app does and how it does it,
without reinforcement-learning terminology. For the full technical version, see
[TUTORIAL.md](TUTORIAL.md).

---

## 1. The problem it solves

Every time someone loads a webpage, an **auction** happens in about 100 milliseconds. Our
system receives a **bid request** that says, roughly:

> "User U is looking at website W, in slot P (say, the header banner). Do you want to show
> an ad? Which one, and what will you pay?"

We have a pool of ads to choose from. Some fit the page (an investment ad on a finance site),
some fit the user (they read gaming media every day), some are just appealing. We want to
**pick the ad most likely to be clicked**, suggest a price, and answer in single-digit
milliseconds.

Two things make this hard:

1. **We learn from history, not experiments.** We can't freely try ads on live traffic —
   mistakes cost real money. So we learn from **past auction logs** that an older, mediocre
   system produced.
2. **The market changes.** New campaigns launch, budgets run out, competitors change bids.
   A model trained once goes stale, so the system **retrains itself every few hours**.

## 2. The core idea

We train a **model that scores every candidate ad** for the current request and tells us which
one is most likely to be clicked. At request time we run that model once, pick the
highest-scoring ad, and suggest a price for it.

That's the whole app. The rest is how the model is trained and kept up to date.

## 3. What the model looks at (the "context")

For each request, the model is given a description of the situation, including:

- **The website** — what kind of content it is (finance, gaming, news…), represented as a
  numeric vector.
- **The user** — what they tend to read, derived from their past engagement.
- **How well the user and site match** — a finance reader on a finance site is a better fit.
- **The site's general click rates** — some sites just get more clicks.
- **Market stats** — how often we tend to win auctions on this site/slot and at what price.
- **The placement** — header vs. sidebar vs. footer behave differently.

## 4. How each candidate ad is described

Each ad is also turned into a set of numbers:

- **Its content** — what the ad text is about (a numeric vector).
- **How well it matches the site and the user.**
- **Its historical click rate** (smoothed so brand-new ads aren't treated as zero forever).
- **Its bid cap** — the most the advertiser allows us to bid.
- **A small learned "identity" vector** per known ad — a learned number that captures
  everything *not* explained by the content (brand strength, creative quality, price point).
  Two ads with similar text can perform very differently, and this lets the model learn that.

The model takes the **context + one ad's description** and outputs a single **score**: how
clickable this ad is in this situation. We repeat this for every candidate ad in one batch and
pick the highest score.

## 5. How the model is trained

Training looks like ordinary supervised learning: for each past auction we *won* and where the
ad was actually shown, we know the context, which ad was shown, and whether it was clicked. We
nudge the model so its score for "this ad in this context" moves toward what actually happened
(1 if clicked, 0 if not, minus a tiny penalty for the price paid).

A few honest points:

- **We only train on auctions we won**, because a click is only observable if the ad was
  actually shown. For lost auctions we have no idea whether the user would have clicked.
- The model is kept **conservative about ads it has little evidence for**, so it doesn't
  confidently pick an ad just because it has no data to say it's bad. (The full tutorial calls
  this "CQL"; the details are technical and not needed to understand the app.)
- Today the training treats each impression as an **independent decision** — it does not yet
  plan ahead across a campaign's daily budget. The machinery for that exists in the code but is
  not switched on in production training.

## 6. Brand-new ads and users (cold start)

- **New ad** (launched an hour ago, zero history): it has no learned identity vector and no
  click history. We **borrow from lookalikes** — find the 5 most similar existing ads by
  content, and give the new ad the average of their identity vectors and click-rate priors. The
  model scores it sensibly on its very first impression, with no retraining. As real history
  accrues, the borrowed values are gradually replaced by actual data.
- **New user** (no engagement history): we assume they're aligned with what they're reading
  *right now* and use the website's description as a stand-in. This prior fades automatically as
  their engagement history grows.

## 7. Serving a request (the fast path)

When a bid request arrives ([`serving/inference.py`](src/rtb_rl/serving/inference.py)):

1. Build the context from the in-memory feature snapshot (no database call on the hot path).
2. Build the description for every eligible ad, including cold-start ads.
3. Run the model once over all candidates → get a score for each → **pick the highest score**.
4. Suggest a price: bid more the more valuable the impression looks, never below the floor or
   above the ad's bid cap. (This is a deliberately simple rule — see the note below.)

Because the model and features live in process memory, the whole path is sub-millisecond on
CPU for ~40 candidates.

> **Honest note on pricing:** the suggested price is a simple rule, **not** a learned bid
> optimizer. It does *not* try to "bid the least possible to win the auction" or pace the
> budget across the day. Those are separate, harder problems (bid shading and budget pacing)
> that this PoC deliberately leaves out. The model only learns **which ad to show**; the price
> is a hand-written heuristic layered on top.

## 8. Retraining: keeping up with a moving market

Every few hours ([`pipelines/retrain_loop.py`](src/rtb_rl/pipelines/retrain_loop.py)):

```
rebuild features from fresh logs → continue training from the current model
      → test the new model in a simulator → only promote it if it's clearly better
      → serving swaps to the new model with no restart
```

- **Continue from current weights** so each cycle adapts incrementally instead of starting over.
- **Simulator gate**: a new model must show higher click-through in the offline simulator before
  it's allowed to go live.
- **Hot swap**: serving notices the new model and reloads it in the background, with no
  downtime.

## 9. One request, end to end

A concrete trace using the demo's synthetic world:

> Request: user `u000123` (reads finance and tech media), website `w0007`
> (`finance007.example.jp`), placement `header`.

1. **Context**: `w0007` is a finance site; `u000123` reads finance — high fit. The site's base
   click rate and the market stats for `(w0007, header)` are attached.
2. **Candidates**: 40 eligible ads, including `ad117` — a brand-new investment campaign with
   zero history, scored using its 5 most similar known finance ads.
3. **Scoring**: the model scores all 40 in one pass. Finance ads score high (content match ×
   user fit), `ad117` lands near its lookalikes, an unrelated discount ad scores low.
4. **Decision**: highest score → say, `ad031` (an established finance ad), with a suggested
   bid. The response returns in ~1 ms, including the model version and a `cold_start` flag.
5. **Later**: tonight's retrain folds the new logs in, continues training, and — if the
   simulator says the new model is better — tomorrow's requests use it.

## 10. Where to read next

- [README.md](README.md) — quickstart, commands, and results.
- [ARCHITECTURE.md](ARCHITECTURE.md) — full component map and the honest list of known gaps.
- [TUTORIAL.md](TUTORIAL.md) — the full version with all the reinforcement-learning details.