"""Synthetic ad-tech data with a *learnable* latent click model.

The generator and the offline simulator share the same :class:`LatentClickModel`, so the
"ground truth" used to label historical clicks is the same signal the simulator uses to score
a learned policy. The latent structure is deliberately recoverable from the features the model
gets to see:

- ``vertical_match``        ~ cosine(ad_embedding, website_embedding)   (both embed their topic)
- ``user_category_affinity`` ~ cosine(user_embedding, ad_embedding) / stored affinity
- ``placement`` / ``website.base_ctr`` are passed to the state directly
- ``ad_appeal``            is a per-ad residual — the part that rewards *learning* and the
                            every-N-hours retraining as inventory/market shifts.

A handful of ads are flagged ``is_coldstart`` and withheld from the historical logs entirely,
so the cold-start path can be exercised end-to-end.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from rtb_rl.config import Config
from rtb_rl.schemas import Ad, BidLog, User, Website

# --- Japanese content templates per vertical (kept short; enough to drive embeddings) ---
_VERTICAL_JP: dict[str, tuple[str, str]] = {
    "finance": ("資産運用と投資の最新情報", "株式・投資信託・NISA・iDeCoなど資産形成を解説するメディア"),
    "ecommerce": ("通販でお得にお買い物", "家電・ファッション・日用品をまとめて比較できるショッピングサイト"),
    "news": ("最新ニュースと速報", "政治・経済・社会・国際の話題を毎日配信するニュースサイト"),
    "gaming": ("ゲーム攻略と最新情報", "新作ゲームのレビュー・攻略・eスポーツ情報を扱うゲームメディア"),
    "travel": ("旅行とホテル予約", "国内・海外旅行の格安プランや観光スポットを紹介する旅行サイト"),
    "health": ("健康と医療の情報", "ダイエット・サプリ・病気予防など健康に関する総合情報サイト"),
    "tech": ("テクノロジーとガジェット", "スマホ・PC・AI・最新ガジェットのレビューを届けるテックメディア"),
    "food": ("グルメとレシピ", "人気レストランや簡単レシピ、お取り寄せグルメを紹介する食メディア"),
    "sports": ("スポーツ速報と分析", "野球・サッカー・バスケなどの試合結果と分析を配信するサイト"),
    "entertainment": ("エンタメと芸能ニュース", "映画・音楽・ドラマ・芸能の最新情報をお届けするエンタメサイト"),
}

_CATEGORY_CREATIVE_JP: dict[str, str] = {
    "finance": "今なら口座開設で手数料無料。かんたん資産運用を始めよう",
    "ecommerce": "期間限定セール開催中。人気商品が最大50%オフ",
    "news": "話題のニュースをアプリで。無料ダウンロードはこちら",
    "gaming": "新作RPG配信開始。今すぐ無料でプレイ",
    "travel": "国内ホテルが今だけ特別価格。週末の旅を予約しよう",
    "health": "初回限定980円。話題のサプリでスッキリ習慣",
    "tech": "最新スマホ予約受付中。下取りでさらにお得に",
    "food": "人気店の味をご自宅へ。送料無料でお取り寄せ",
    "sports": "プロの試合を見放題。スポーツ配信を無料体験",
    "entertainment": "話題の映画が見放題。30日間無料トライアル",
}

# Additive logit contribution of each placement (visibility vs. intrusiveness trade-off).
_PLACEMENT_LOGIT: dict[str, float] = {
    "header": 0.6,
    "in_article": 0.4,
    "interstitial": 0.5,
    "sidebar": -0.1,
    "footer": -0.5,
}
# Typical clearing-price level (JPY) per placement; lognormal mean.
_PLACEMENT_PRICE: dict[str, float] = {
    "header": 120.0,
    "in_article": 90.0,
    "interstitial": 150.0,
    "sidebar": 60.0,
    "footer": 40.0,
}

_ADVERTISERS = [
    "Sakura Inc", "Fuji Corp", "Akari Co", "Hikari Ltd", "Tsubaki KK",
    "Midori Holdings", "Aoi Group", "Kiri Media", "Hoshi Tech", "Yume Brands",
]


@dataclass
class LatentClickModel:
    """Ground-truth click probability shared by the generator and the simulator."""

    base_logit: float
    placement_logit: dict[str, float]
    website_quality: dict[str, float]  # in [0, 1]
    ad_appeal: dict[str, float]  # centered residual, ~N(0, 0.5)
    # Per-website vertical and per-user engagement are needed to compute affinities.
    website_vertical: dict[str, str]
    ad_category: dict[str, str]
    ad_targets: dict[str, list[str]]
    user_cat_strength: dict[str, dict[str, float]]  # user_id -> {vertical -> [0,1]}

    def user_category_affinity(self, user_id: str, category: str) -> float:
        return self.user_cat_strength.get(user_id, {}).get(category, 0.0)

    def vertical_match(self, ad_id: str, website_id: str) -> float:
        vert = self.website_vertical[website_id]
        if self.ad_category[ad_id] == vert:
            return 1.0
        if vert in self.ad_targets.get(ad_id, []):
            return 0.4
        return 0.0

    def click_logit(self, user_id: str, ad_id: str, website_id: str, placement: str) -> float:
        return (
            self.base_logit
            + 2.0 * self.vertical_match(ad_id, website_id)
            + 2.5 * self.user_category_affinity(user_id, self.ad_category[ad_id])
            + self.placement_logit.get(placement, 0.0)
            + 3.0 * (self.website_quality[website_id] - 0.5)
            + self.ad_appeal[ad_id]
        )

    def p_click(self, user_id: str, ad_id: str, website_id: str, placement: str) -> float:
        return 1.0 / (1.0 + math.exp(-self.click_logit(user_id, ad_id, website_id, placement)))

    # ---- (de)serialization ----
    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> LatentClickModel:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass
class SyntheticDataset:
    websites: list[Website]
    users: list[User]
    ads: list[Ad]  # includes cold-start ads (which never appear in bid_logs)
    bid_logs: list[BidLog]
    latent: LatentClickModel
    extras: dict = field(default_factory=dict)


def generate(cfg: Config) -> SyntheticDataset:
    """Generate a full synthetic dataset deterministically from ``cfg.seed``."""
    rng = np.random.default_rng(cfg.seed)
    d = cfg.data

    # ---- websites ----
    websites: list[Website] = []
    website_quality: dict[str, float] = {}
    website_vertical: dict[str, str] = {}
    for i in range(d.n_websites):
        vert = d.verticals[i % len(d.verticals)]
        title, desc = _VERTICAL_JP.get(vert, (vert, vert))
        quality = float(np.clip(rng.beta(2.5, 2.5), 0.02, 0.98))
        base_ctr = float(np.clip(0.005 + 0.06 * quality + rng.normal(0, 0.005), 0.001, 0.2))
        wid = f"w{i:04d}"
        websites.append(
            Website(
                website_id=wid,
                domain=f"{vert}{i:03d}.example.jp",
                vertical=vert,
                title=f"{title}｜{vert}{i:03d}",
                description=desc,
                base_ctr=round(base_ctr, 5),
                base_cvr=round(float(np.clip(base_ctr * rng.uniform(0.1, 0.4), 0.0005, 0.1)), 5),
                avg_daily_impressions=int(rng.integers(5_000, 500_000)),
            )
        )
        website_quality[wid] = quality
        website_vertical[wid] = vert

    by_vertical: dict[str, list[str]] = {v: [] for v in d.verticals}
    for w in websites:
        by_vertical[w.vertical].append(w.website_id)

    # ---- users + engagement ----
    users: list[User] = []
    user_cat_strength: dict[str, dict[str, float]] = {}
    for u in range(d.n_users):
        uid = f"u{u:06d}"
        n_pref = int(rng.integers(1, 3))
        prefs = list(rng.choice(d.verticals, size=n_pref, replace=False))
        n_sites = int(rng.integers(1, 8))
        engagement: dict[str, int] = {}
        for _ in range(n_sites):
            # 80% of visited sites come from a preferred vertical.
            vert = (
                str(rng.choice(prefs))
                if rng.random() < 0.8
                else str(rng.choice(d.verticals))
            )
            if not by_vertical[vert]:
                continue
            wid = str(rng.choice(by_vertical[vert]))
            # Higher engagement on preferred verticals.
            hi = vert in prefs
            level = int(np.clip(rng.integers(4, 11) if hi else rng.integers(0, 6), 0, 10))
            engagement[wid] = max(engagement.get(wid, 0), level)
        users.append(User(user_id=uid, engagement=engagement))

        # Per-user category strength = engagement mass per vertical (normalized).
        strength: dict[str, float] = {}
        total = 0.0
        for wid, lvl in engagement.items():
            v = website_vertical[wid]
            strength[v] = strength.get(v, 0.0) + lvl
            total += lvl
        if total > 0:
            strength = {v: s / total for v, s in strength.items()}
        user_cat_strength[uid] = strength

    # ---- ads / campaigns (some cold-start, withheld from logs) ----
    ads: list[Ad] = []
    ad_appeal: dict[str, float] = {}
    ad_category: dict[str, str] = {}
    ad_targets: dict[str, list[str]] = {}
    n_cold = min(d.n_coldstart_ads, max(0, d.n_ads - 1))
    for a in range(d.n_ads):
        cat = d.verticals[int(rng.integers(0, len(d.verticals)))]
        campaign = f"cmp{int(rng.integers(0, d.n_campaigns)):02d}"
        adv = _ADVERTISERS[int(rng.integers(0, len(_ADVERTISERS)))]
        targets = [cat]
        if rng.random() < 0.4:
            targets.append(str(rng.choice(d.verticals)))
        appeal = float(rng.normal(0.0, 0.5))
        aid = f"ad{a:03d}"
        is_cold = a >= (d.n_ads - n_cold)
        ads.append(
            Ad(
                ad_id=aid,
                campaign_id=campaign,
                advertiser=adv,
                category=cat,
                creative_text=_CATEGORY_CREATIVE_JP.get(cat, cat),
                bid_cap_jpy=round(float(rng.uniform(50, 300)), 1),
                daily_budget_jpy=round(float(rng.uniform(5_000, 50_000)), 0),
                target_verticals=sorted(set(targets)),
                is_coldstart=is_cold,
            )
        )
        ad_appeal[aid] = appeal
        ad_category[aid] = cat
        ad_targets[aid] = sorted(set(targets))

    latent = LatentClickModel(
        base_logit=-3.0,
        placement_logit=_PLACEMENT_LOGIT,
        website_quality=website_quality,
        ad_appeal=ad_appeal,
        website_vertical=website_vertical,
        ad_category=ad_category,
        ad_targets=ad_targets,
        user_cat_strength=user_cat_strength,
    )

    # ---- historical bid logs (behaviour policy = mostly-random ad selection) ----
    loggable_ads = [ad for ad in ads if not ad.is_coldstart]
    bid_logs = _generate_bid_logs(cfg, rng, websites, users, loggable_ads, latent)

    return SyntheticDataset(
        websites=websites, users=users, ads=ads, bid_logs=bid_logs, latent=latent
    )


def _generate_bid_logs(
    cfg: Config,
    rng: np.random.Generator,
    websites: list[Website],
    users: list[User],
    ads: list[Ad],
    latent: LatentClickModel,
) -> list[BidLog]:
    d = cfg.data
    placements = d.placements
    start = datetime(2024, 1, 1)
    ad_arr = np.array([a.ad_id for a in ads])
    bid_caps = {a.ad_id: a.bid_cap_jpy for a in ads}
    logs: list[BidLog] = []
    for i in range(d.n_bid_logs):
        w = websites[int(rng.integers(0, len(websites)))]
        u = users[int(rng.integers(0, len(users)))]
        placement = str(rng.choice(placements))
        # Behaviour policy: pick an ad almost uniformly (good action coverage for offline RL),
        # with a mild bias toward category-matching ads to look like a weak production policy.
        if rng.random() < 0.5:
            cands = [a.ad_id for a in ads if a.category == w.vertical] or list(ad_arr)
            ad_id = str(rng.choice(cands))
        else:
            ad_id = str(rng.choice(ad_arr))

        market = float(np.clip(rng.lognormal(math.log(_PLACEMENT_PRICE[placement]), 0.4), 5, 2000))
        cap = bid_caps[ad_id]
        bid = float(min(cap, np.clip(market * rng.uniform(0.6, 1.4), 5, 2000)))
        won = bid >= market
        click = bool(won and rng.random() < latent.p_click(u.user_id, ad_id, w.website_id, placement))
        logs.append(
            BidLog(
                request_id=f"r{i:08d}",
                ts=start + timedelta(seconds=int(i * 2)),
                website_id=w.website_id,
                placement=placement,
                user_id=u.user_id,
                ad_id=ad_id,
                bid_price_jpy=round(bid, 2),
                market_price_jpy=round(market, 2),
                won=won,
                click=click,
                cost_jpy=round(market, 2) if won else 0.0,
            )
        )
    return logs
