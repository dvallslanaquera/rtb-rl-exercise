"""Domain schemas (Pydantic v2) shared across data, features, RL, sim and serving.

These mirror the shape of data a classic ad-tech stack would expose:
- :class:`Website` — publisher inventory context (vertical, traffic-quality metrics, JP text),
- :class:`User`    — a unique id with sparse per-site engagement levels (0..10),
- :class:`Ad`      — a creative belonging to a campaign, with budget/pacing,
- :class:`BidLog`  — one historical auction record (placement, price in JPY, win, click),
- :class:`BidRequest` / :class:`BidResponse` — the real-time serving contract.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# Engagement scale: 0 = landing-page bounce only ... 10 = browsed, listed products, purchased.
MIN_ENGAGEMENT = 0
MAX_ENGAGEMENT = 10


class Website(BaseModel):
    website_id: str
    domain: str
    vertical: str
    title: str  # JP text
    description: str  # JP text
    base_ctr: float  # historical click-through rate of the site
    base_cvr: float  # historical conversion rate
    avg_daily_impressions: int

    def content_text(self) -> str:
        """Concatenated text fed to the embedder."""
        return f"{self.vertical} {self.title} {self.description}".strip()


class User(BaseModel):
    user_id: str
    # Sparse mapping website_id -> engagement level in [0, 10].
    engagement: dict[str, int] = Field(default_factory=dict)


class Ad(BaseModel):
    ad_id: str
    campaign_id: str
    advertiser: str
    category: str  # aligns with website verticals
    creative_text: str  # JP text
    bid_cap_jpy: float  # max the campaign will pay per impression
    daily_budget_jpy: float
    target_verticals: list[str] = Field(default_factory=list)
    is_coldstart: bool = False  # True for ads withheld from the historical logs

    def content_text(self) -> str:
        return f"{self.category} {self.creative_text}".strip()


class BidLog(BaseModel):
    request_id: str
    ts: datetime
    website_id: str
    placement: str
    user_id: str
    ad_id: str  # ad that was bid on / served
    bid_price_jpy: float
    market_price_jpy: float  # clearing/second price (used by the offline simulator)
    won: bool
    click: bool
    cost_jpy: float  # price actually paid (0 if lost)


class BidRequest(BaseModel):
    """Incoming real-time bid opportunity."""

    request_id: str
    website_id: str
    placement: str
    user_id: str
    # Optional explicit candidate set; when omitted, eligible ads are selected server-side.
    candidate_ad_ids: list[str] | None = None
    # Optional auction floor. When omitted (None) the server applies cfg.serving.default_floor_jpy;
    # an explicit 0.0 is honored as "no floor". The None default is what lets the configured
    # fallback actually take effect on the hot path.
    floor_price_jpy: float | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BidResponse(BaseModel):
    """Server decision for a :class:`BidRequest`."""

    request_id: str
    ad_id: str  # chosen advertisement
    bid_price_jpy: float  # suggested bid
    predicted_click_value: float  # Q-value of the chosen (state, ad)
    model_version: str
    cold_start: bool = False  # chosen ad/user/site relied on a cold-start prior
    latency_ms: float = 0.0
