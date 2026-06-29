from rtb_rl.data.synth import generate


def test_generate_counts(cfg):
    ds = generate(cfg)
    assert len(ds.websites) == 12
    assert len(ds.users) == 80
    assert len(ds.ads) == 20
    assert sum(a.is_coldstart for a in ds.ads) == 3
    assert len(ds.bid_logs) == 2500


def test_coldstart_ads_withheld_from_logs(cfg):
    ds = generate(cfg)
    cold = {a.ad_id for a in ds.ads if a.is_coldstart}
    logged = {b.ad_id for b in ds.bid_logs}
    assert cold.isdisjoint(logged)


def test_pclick_in_range_and_deterministic(cfg):
    a = generate(cfg)
    b = generate(cfg)
    log = a.bid_logs[0]
    p = a.latent.p_click(log.user_id, log.ad_id, log.website_id, log.placement)
    assert 0.0 <= p <= 1.0
    # same seed -> identical click totals
    assert sum(x.click for x in a.bid_logs) == sum(x.click for x in b.bid_logs)


def test_some_clicks_exist(cfg):
    ds = generate(cfg)
    assert sum(b.click for b in ds.bid_logs) > 0
    assert sum(b.won for b in ds.bid_logs) > 0
