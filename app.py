"""Streamlit dashboard: crypto news & Reddit sentiment vs. price.

Run locally:  streamlit run app.py
"""

import html
import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import COINS, META_JSON, POSTS_CSV, PRICES_CSV
from src.sentiment import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD

st.set_page_config(page_title="Crypto Sentiment Dashboard", page_icon="📈", layout="wide")
st.html(Path(__file__).parent / "assets" / "theme.css")

LABELS = ["positive", "neutral", "negative"]
SOURCES = {"All sources": ["news", "reddit"], "News": ["news"], "Reddit": ["reddit"]}
PERIODS = {"7D": 7, "30D": 30, "90D": 90}
MIN_DAYS_FOR_CORRELATION = 10
# A daily average from one or two headlines is noise, not signal.
MIN_ITEMS_PER_DAY = 5
LOW_SAMPLE_OPACITY = 0.3
MIN_ITEMS_FOR_COMPARISON = 20
REPO_URL = "https://github.com/Z3US-23/crypto-dashboard-v2"

# Gold means up (rising price, positive mood) and pink means down, everywhere on the page.
UP, DOWN, NEUTRAL = "#f6c56f", "#f0569c", "#7c84b8"
MUTED, FAINT = "#9aa1cf", "#646b9e"
LABEL_COLORS = {"positive": UP, "neutral": NEUTRAL, "negative": DOWN}
COIN_STYLE = {
    "BTC": {"color": "#f7931a", "icon": "#f7931a", "glyph": "₿"},
    "ETH": {"color": "#8b9bff", "icon": "#627eea", "glyph": "Ξ"},
    "SOL": {"color": "#3ee0b4", "icon": "linear-gradient(135deg, #9945ff, #14f195)", "glyph": "◎"},
}


@st.cache_data(ttl=600)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    posts = pd.read_csv(POSTS_CSV)
    posts["published_at"] = pd.to_datetime(posts["published_at"], utc=True, format="ISO8601")
    posts["date"] = posts["published_at"].dt.tz_convert(None).dt.normalize()

    prices = pd.read_csv(PRICES_CSV, parse_dates=["date"]).sort_values(["coin", "date"])
    prices["next_day_return"] = prices.groupby("coin")["close"].transform(lambda s: s.shift(-1) / s - 1)

    meta = json.loads(META_JSON.read_text()) if META_JSON.exists() else {}
    return posts, prices, meta


def daily_sentiment(posts: pd.DataFrame) -> pd.DataFrame:
    daily = (
        posts.groupby(["coin", "date"])
        .agg(mentions=("id", "size"), avg_sentiment=("sentiment", "mean"))
        .reset_index()
    )
    counts = (
        posts.groupby(["coin", "date", "label"]).size()
        .unstack(fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
        .reset_index()
    )
    return daily.merge(counts, on=["coin", "date"])


def time_ago(when) -> str:
    if not isinstance(when, pd.Timestamp) and not when:
        return "unknown"
    minutes = max(int((pd.Timestamp.now(tz="UTC") - pd.Timestamp(when)).total_seconds() // 60), 0)
    if minutes < 60:
        return f"{minutes} min ago"
    if minutes < 48 * 60:
        return f"{minutes // 60} h ago"
    return f"{minutes // (24 * 60)} days ago"


def mood_of(value: float) -> str:
    if value >= POSITIVE_THRESHOLD:
        return "positive"
    if value <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def spread_labels(values: dict[str, float], min_gap: float) -> dict[str, float]:
    """Nudge end-of-line label positions apart so they never overlap."""
    placed: dict[str, float] = {}
    previous = None
    for key, value in sorted(values.items(), key=lambda item: item[1]):
        position = value if previous is None else max(value, previous + min_gap)
        placed[key] = previous = position
    return placed


def render(markup: str) -> None:
    """Show raw HTML. '$' is escaped so Streamlit doesn't read two prices as a LaTeX formula."""
    st.markdown(markup.replace("$", "&#36;"), unsafe_allow_html=True)


def money(value: float) -> str:
    return f"${value:,.2f}"


def coin_icon(symbol: str, size: int = 26) -> str:
    style = COIN_STYLE[symbol]
    return (
        f'<span class="coin-ico" style="background:{style["icon"]};width:{size}px;height:{size}px;'
        f'font-size:{round(size * 0.55)}px">{style["glyph"]}</span>'
    )


def change_badge(change: float) -> str:
    direction, arrow = ("up", "▲") if change >= 0 else ("down", "▼")
    return f'<span class="badge {direction}">{arrow} {abs(change):.2%}</span>'


def sparkline(values: list[float], color: str, uid: str, width: int = 240, height: int = 64) -> str:
    """Smooth SVG line that fades in from the left and ends in a ringed dot."""
    if len(values) < 2:
        return ""
    low, high, pad = min(values), max(values), 6
    span = (high - low) or 1
    points = [
        (pad + i * (width - 2 * pad) / (len(values) - 1), pad + (1 - (v - low) / span) * (height - 2 * pad))
        for i, v in enumerate(values)
    ]
    # Catmull-Rom spline through every point, written as cubic Bezier segments.
    path = f"M{points[0][0]:.1f},{points[0][1]:.1f}"
    for i in range(len(points) - 1):
        p0, p1, p2 = points[max(i - 1, 0)], points[i], points[i + 1]
        p3 = points[min(i + 2, len(points) - 1)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        path += f" C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}"
    end_x, end_y = points[-1]
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}">'
        f'<defs><linearGradient id="{uid}" x1="0" x2="1" y1="0" y2="0">'
        f'<stop offset="0" stop-color="{color}" stop-opacity="0"/>'
        f'<stop offset="0.5" stop-color="{color}" stop-opacity="0.35"/>'
        f'<stop offset="1" stop-color="{color}"/></linearGradient></defs>'
        f'<path d="{path}" fill="none" stroke="url(#{uid})" stroke-width="2.2" stroke-linecap="round"/>'
        f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="4.5" fill="#0b0f2a" stroke="{color}" stroke-width="2.2"/></svg>'
    )


def price_ticks(low: float, high: float, avoid: float, count: int = 8) -> list[float]:
    """Round-number axis ticks, minus any that would sit under the current-price tag."""
    raw_step = (high - low) / count
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw_step)
    ticks = [math.ceil(low / step) * step + i * step for i in range(count + 2)]
    return [t for t in ticks if low <= t <= high and abs(t - avoid) > step * 0.4]


def gauge(value: float) -> str:
    position = (max(-1.0, min(1.0, value)) + 1) / 2 * 100
    scale = "".join(f"<span>{tick}</span>" for tick in ["−1", "−0.5", "0", "+0.5", "+1"])
    return (
        f'<div class="gauge"><div class="gauge-track"><span class="gauge-knob" style="left:{position:.1f}%"></span>'
        f'</div><div class="gauge-scale">{scale}</div></div>'
    )


def shorten(text: str, limit: int = 110) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def headline_list(items: pd.DataFrame, title: str, color: str) -> str:
    rows = "".join(
        f'<a class="hl" href="{html.escape(row.url)}" target="_blank" rel="noopener">'
        f'<span class="hl-score {mood_of(row.sentiment)}">{row.sentiment:+.2f}</span>'
        f'<span><span class="hl-title">{html.escape(shorten(row.title))}</span>'
        f'<span class="hl-meta">{html.escape(row.source)} · {time_ago(row.published_at)}</span></span></a>'
        for row in items.itertuples()
    )
    return f'<div class="hl-head"><i style="background:{color}"></i>{title}</div>{rows}'


def style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color=MUTED),
        hoverlabel=dict(
            bgcolor="#ffffff", bordercolor="#ffffff", font=dict(family="Inter, sans-serif", size=12, color="#0b0f2a")
        ),
        hovermode="x unified",
        showlegend=False,
        bargap=0.35,
        barcornerradius=3,
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, showline=False, ticks="", tickfont_color=FAINT,
        showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot",
        spikecolor="rgba(255,255,255,0.35)", spikethickness=1,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.045)", zeroline=False, ticks="", tickfont_color=FAINT)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False}, width="stretch")


posts, prices, meta = load_data()
has_ohlc = {"open", "high", "low"}.issubset(prices.columns)

# ---------- Top bar: brand, coin switcher, data status ----------
with st.container(key="topbar"):
    brand, nav, status = st.columns([1.2, 1.6, 1.2], vertical_alignment="center")
    with brand:
        render('<div class="brand"><span class="brand-ring"></span>Crypto Sentiment</div>')
    with nav:
        coin = st.segmented_control(
            "Coin", list(COINS), default="BTC", key="coin", label_visibility="collapsed",
            format_func=lambda s: f"{COIN_STYLE[s]['glyph']}  {COINS[s]['name']}",
        ) or "BTC"
    with status:
        render(
            f'<div class="status"><span class="chip"><span class="live-dot"></span>'
            f'Updated {time_ago(meta.get("last_run"))}</span>'
            f'<a class="chip" href="{REPO_URL}" target="_blank" rel="noopener">GitHub ↗</a></div>'
        )

render(
    '<p class="intro">What crypto news and Reddit are saying about <b>Bitcoin, Ethereum and Solana</b>, '
    f"and whether the mood lines up with the price. <b>{len(posts):,}</b> headlines and posts scored so far, "
    "collected automatically every 6 hours.</p>"
)

# ---------- Filters live inside the cards they belong to; read them before drawing anything ----------
chart_col, panel_col = st.columns([2.6, 1], gap="medium")
with panel_col:
    panel = st.container(key="card_panel")
    with panel:
        render(f'<div class="panel-title"><div class="card-title">Market mood</div><span>{COINS[coin]["name"]}</span></div>')
        source_label = st.segmented_control(
            "Sources", list(SOURCES), default="All sources", key="sources", label_visibility="collapsed"
        ) or "All sources"
with chart_col:
    chart_card = st.container(key="card_chart")
    with chart_card:
        price_head, period_box = st.columns([3, 1.2])
        with period_box:
            period_label = st.segmented_control(
                "Period", list(PERIODS), default="30D", key="period", label_visibility="collapsed"
            ) or "30D"

days = PERIODS[period_label]
end = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
start = end - pd.Timedelta(days=days - 1)
prev_start = start - pd.Timedelta(days=days)

source_filter = posts["source_type"].isin(SOURCES[source_label])
in_period = posts[source_filter & posts["date"].between(start, end)]
prev_period = posts[source_filter & posts["date"].between(prev_start, start - pd.Timedelta(days=1))]

coin_posts = in_period[in_period["coin"] == coin]
coin_prev = prev_period[prev_period["coin"] == coin]
coin_prices = prices[(prices["coin"] == coin) & prices["date"].between(start, end)]
daily_all = daily_sentiment(in_period) if not in_period.empty else pd.DataFrame()
coin_daily = daily_all[daily_all["coin"] == coin] if not daily_all.empty else pd.DataFrame()
reliable_all = daily_all[daily_all["mentions"] >= MIN_ITEMS_PER_DAY] if not daily_all.empty else daily_all
bar_opacity = [1.0 if n >= MIN_ITEMS_PER_DAY else LOW_SAMPLE_OPACITY for n in coin_daily.get("mentions", [])]
coin_name = COINS[coin]["name"]

if coin_posts.empty:
    with chart_card:
        st.warning("No posts for this coin, period and source selection yet.")
    st.stop()

# ---------- Main chart: daily candles with the day's mood underneath ----------
with price_head:
    if len(coin_prices) >= 2:
        first_close, last_close = coin_prices["close"].iloc[0], coin_prices["close"].iloc[-1]
        render(
            f'<div class="pair">{coin_icon(coin, 30)}{coin} / USD<span class="pair-sub">{coin_name}</span></div>'
            f'<div class="price-row"><span class="price">{money(last_close)}</span>'
            f'{change_badge(last_close / first_close - 1)}<span class="price-note">over {days} days</span></div>'
        )

with chart_card:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.74, 0.26])
    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=coin_prices["date"], open=coin_prices["open"], high=coin_prices["high"],
                low=coin_prices["low"], close=coin_prices["close"],
                increasing=dict(line=dict(color=UP, width=1.3), fillcolor=UP),
                decreasing=dict(line=dict(color=DOWN, width=1.3), fillcolor=DOWN),
                whiskerwidth=0, name="Price", hoverinfo="text",
                text=[
                    f"Open {money(o)}<br>High {money(h)}<br>Low {money(lo)}<br>Close {money(c)}"
                    for o, h, lo, c in coin_prices[["open", "high", "low", "close"]].itertuples(index=False)
                ],
            ),
            row=1, col=1,
        )
    else:  # older prices.csv with closing prices only
        fig.add_trace(
            go.Scatter(
                x=coin_prices["date"], y=coin_prices["close"], mode="lines", line=dict(color=UP, width=2),
                hovertemplate="Close $%{y:,.2f}<extra></extra>", name="Price",
            ),
            row=1, col=1,
        )
    if not coin_prices.empty:
        latest = coin_prices["close"].iloc[-1]
        fig.add_hline(y=latest, line=dict(color="rgba(255,255,255,0.45)", width=1, dash="dash"), row=1, col=1)
        fig.add_annotation(
            x=0, xref="x domain", y=latest, yref="y", xanchor="right", showarrow=False,
            text=f"{latest:,.0f}" if latest >= 1000 else f"{latest:,.2f}",
            bgcolor="#ffffff", borderpad=4, font=dict(color="#0b0f2a", size=11),
        )
        low = coin_prices["low" if has_ohlc else "close"].min()
        high = coin_prices["high" if has_ohlc else "close"].max()
        pad = (high - low) * 0.06 or high * 0.01
        ticks = price_ticks(low - pad, high + pad, latest)
        fig.update_yaxes(
            range=[low - pad, high + pad], tickvals=ticks,
            ticktext=[f"${t / 1000:g}k" if t >= 1000 else f"${t:g}" for t in ticks], row=1, col=1,
        )
    fig.add_trace(
        go.Bar(
            x=coin_daily["date"], y=coin_daily["avg_sentiment"],
            marker=dict(color=[LABEL_COLORS[mood_of(v)] for v in coin_daily["avg_sentiment"]], opacity=bar_opacity),
            customdata=coin_daily["mentions"],
            hovertemplate="Mood %{y:+.2f} (%{customdata} items)<extra></extra>", name="Mood",
        ),
        row=2, col=1,
    )
    fig.add_annotation(
        x=0, xref="x2 domain", y=1, yref="y2 domain", xanchor="left", yanchor="bottom", showarrow=False,
        text="DAILY MOOD", font=dict(color=FAINT, size=10),
    )
    mood_limit = max(0.3, coin_daily["avg_sentiment"].abs().max() * 1.2)
    fig.update_yaxes(
        range=[-mood_limit, mood_limit], tickformat="+.1f", nticks=5,
        zeroline=True, zerolinecolor="rgba(255,255,255,0.12)", row=2, col=1,
    )
    fig.update_xaxes(range=[start - pd.Timedelta(hours=12), end + pd.Timedelta(hours=12)], rangeslider_visible=False)
    fig.update_xaxes(tickformat="%b %d", row=2, col=1)
    show(style(fig, 540).update_layout(margin=dict(l=64)))
    render(
        '<p class="chart-note">Candles show each day\'s open, high, low and close. Bars show that day\'s average '
        "sentiment: gold when mostly positive, pink when mostly negative. "
        f"Faded bars are days with fewer than {MIN_ITEMS_PER_DAY} items, too few to trust.</p>"
    )

# ---------- Mood panel ----------
avg = coin_posts["sentiment"].mean()
prev_avg = coin_prev["sentiment"].mean() if len(coin_prev) >= MIN_ITEMS_FOR_COMPARISON else None
mood = mood_of(avg)
shares = coin_posts["label"].value_counts(normalize=True).reindex(LABELS, fill_value=0)
by_type = coin_posts["source_type"].value_counts()
coin_reliable = coin_daily[coin_daily["mentions"] >= MIN_ITEMS_PER_DAY]
joined = coin_reliable.merge(prices[prices["coin"] == coin][["date", "next_day_return"]], on="date").dropna()
source_status = meta.get("sources", {})
sources_ok = sum(result.startswith("ok") for result in source_status.values())

split = "".join(f'<span style="flex:{shares[lab]:.3f};background:{LABEL_COLORS[lab]}"></span>' for lab in LABELS)
split_pills = "".join(
    f'<span class="pill{" on" if lab == shares.idxmax() else ""}">{shares[lab]:.0%} {lab}</span>' for lab in LABELS
)
facts = [
    ("Days with enough data", f"{len(coin_reliable)} of {days}"),
    (
        "Mood vs next-day price",
        f"r = {joined['avg_sentiment'].corr(joined['next_day_return']):+.2f}"
        if len(joined) >= MIN_DAYS_FOR_CORRELATION else f"{len(joined)} of {MIN_DAYS_FOR_CORRELATION} days",
    ),
    ("Sources online", f"{sources_ok} of {len(source_status)}"),
    ("Scoring model", "VADER + crypto lexicon"),
]
with panel:
    render(
        f'<div class="well"><div class="well-head" title="VADER compound score from -1 (very negative) to +1 '
        f'(very positive), averaged over every headline and post."><span>Average sentiment</span>'
        f'<span class="tag {mood}">{mood.capitalize()}</span></div>'
        f'<div class="well-value">{avg:+.2f}</div>'
        f'<div class="well-note">'
        + (f"{avg - prev_avg:+.2f} vs previous {days} days" if prev_avg is not None else f"{coin_name}, last {days} days")
        + f"</div>{gauge(avg)}</div>"
        f'<div class="well"><div class="well-head"><span>Mentions</span><span>last {days} days</span></div>'
        f'<div class="well-value">{len(coin_posts):,}</div><div class="pills">'
        f'<span class="pill">News {by_type.get("news", 0):,}</span>'
        f'<span class="pill">Reddit {by_type.get("reddit", 0):,}</span></div></div>'
        f'<div class="well"><div class="well-head"><span>Mood split</span><span>share of items</span></div>'
        f'<div class="split">{split}</div><div class="pills">{split_pills}</div></div>'
        '<a class="cta" href="#headlines" target="_self">Read the headlines</a>'
        + "".join(f'<div class="kv"><span>{k}</span><b>{v}</b></div>' for k, v in facts)
    )

# ---------- Ticker cards: every coin at a glance, plus the overall mood ----------
tickers = st.columns(4, gap="small")
for column, symbol in zip(tickers, COINS):
    closes = prices[(prices["coin"] == symbol) & prices["date"].between(start, end)]["close"].tolist()
    coin_mood = in_period.loc[in_period["coin"] == symbol, "sentiment"].mean()
    if len(closes) < 2:
        continue
    change = closes[-1] / closes[0] - 1
    mood_text = (
        f'Mood <b style="color:{LABEL_COLORS[mood_of(coin_mood)]}">{coin_mood:+.2f}</b>'
        if pd.notna(coin_mood) else "No posts yet"
    )
    with column:
        render(
            f'<div class="ticker{" active" if symbol == coin else ""}" style="--glow:{COIN_STYLE[symbol]["color"]}">'
            f'<div class="ticker-head">{coin_icon(symbol, 24)}{symbol} / USD{change_badge(change)}</div>'
            f'<div class="ticker-price">{money(closes[-1])}</div>'
            f'<div class="ticker-foot"><span class="ticker-mood">{mood_text}</span>'
            f"{sparkline(closes, UP if change >= 0 else DOWN, f'spark-{symbol}')}</div></div>"
        )

market = in_period.groupby("date").agg(mentions=("id", "size"), avg_sentiment=("sentiment", "mean"))
market_trend = market.loc[market["mentions"] >= MIN_ITEMS_PER_DAY, "avg_sentiment"].tolist()
market_avg = in_period["sentiment"].mean()
market_mood = mood_of(market_avg)
with tickers[3]:
    render(
        f'<div class="ticker" style="--glow:#8c6cf0"><div class="ticker-head">'
        f'<span class="coin-ico" style="background:#8c6cf0;width:24px;height:24px;font-size:12px">◐</span>'
        f'All coins<span class="tag {market_mood}">{market_mood.capitalize()}</span></div>'
        f'<div class="ticker-price">{market_avg:+.2f}</div>'
        f'<div class="ticker-foot"><span class="ticker-mood"><b>{len(in_period):,}</b> items</span>'
        f"{sparkline(market_trend, LABEL_COLORS[market_mood], 'spark-market')}</div></div>"
    )

# ---------- Mood mix + coin comparison ----------
mix_col, compare_col = st.columns(2, gap="medium")
with mix_col, st.container(key="card_mix"):
    render(
        '<div class="card-title">Mood mix per day</div>'
        f'<p class="card-deck">Share of positive, neutral and negative {coin_name} items each day. '
        f"Faded: fewer than {MIN_ITEMS_PER_DAY} items.</p>"
    )
    mix = go.Figure()
    for lab in LABELS:
        mix.add_trace(
            go.Bar(
                x=coin_daily["date"], y=coin_daily[lab], name=lab.capitalize(),
                marker=dict(color=LABEL_COLORS[lab], opacity=bar_opacity),
                customdata=coin_daily["mentions"],
                hovertemplate=f"{lab.capitalize()}: %{{y}} of %{{customdata}}<extra></extra>",
            )
        )
    mix.update_layout(barmode="stack", barnorm="percent")
    mix.update_yaxes(ticksuffix="%", range=[0, 100], nticks=5)
    mix.update_xaxes(tickformat="%b %d")
    show(style(mix, 320).update_layout(bargap=0.45))

with compare_col, st.container(key="card_compare"):
    render(
        '<div class="card-title">All three coins compared</div>'
        f'<p class="card-deck">Average daily sentiment per coin, same period and sources. '
        f"Days with fewer than {MIN_ITEMS_PER_DAY} items are left out.</p>"
    )
    compare = go.Figure()
    end_points = {}
    for symbol in COINS:
        series = reliable_all[reliable_all["coin"] == symbol] if not reliable_all.empty else pd.DataFrame()
        if series.empty:
            continue
        end_points[symbol] = (series["date"].iloc[-1], series["avg_sentiment"].iloc[-1])
        # Reindex to every day so missing days show as gaps instead of a line bridging them.
        series = series.set_index("date").reindex(pd.date_range(series["date"].min(), end)).rename_axis("date").reset_index()
        compare.add_trace(
            go.Scatter(
                x=series["date"], y=series["avg_sentiment"], name=COINS[symbol]["name"],
                mode="lines+markers", line=dict(color=COIN_STYLE[symbol]["color"], width=2.2, shape="spline"),
                marker=dict(size=7, line=dict(width=2, color="#0b0f2a")),
                hovertemplate=f"{COINS[symbol]['name']}: %{{y:+.2f}}<extra></extra>",
            )
        )
    if end_points:
        y_range = reliable_all["avg_sentiment"].max() - reliable_all["avg_sentiment"].min()
        label_y = spread_labels({s: y for s, (_, y) in end_points.items()}, min_gap=max(y_range, 0.1) * 0.09)
        for symbol, (x, _) in end_points.items():
            compare.add_annotation(
                x=x, y=label_y[symbol], text=COINS[symbol]["name"], xanchor="left", xshift=10, showarrow=False,
                font=dict(size=12, color=COIN_STYLE[symbol]["color"]),
            )
    compare.update_yaxes(tickformat="+.2f", zeroline=True, zerolinecolor="rgba(255,255,255,0.12)")
    compare.update_xaxes(tickformat="%b %d")
    show(style(compare, 320).update_layout(margin=dict(r=80)))

# ---------- Does sentiment lead price? ----------
with st.container(key="card_corr"):
    render(
        '<div class="card-title">Does today\'s mood predict tomorrow\'s price?</div>'
        f'<p class="card-deck">Each day with at least {MIN_ITEMS_PER_DAY} {coin_name} items is paired with the '
        "price change on the following day.</p>"
    )
    if len(joined) < MIN_DAYS_FOR_CORRELATION:
        render(
            f'<div class="progress"><span style="width:{len(joined) / MIN_DAYS_FOR_CORRELATION:.0%}"></span></div>'
            f'<div class="progress-meta"><span><b>{len(joined)}</b> of {MIN_DAYS_FOR_CORRELATION} days collected</span>'
            "<span>new data every 6 hours</span></div>"
            '<p class="chart-note" style="margin-top:10px">Not enough history for an honest answer yet. '
            "The test switches on by itself once there are enough days with a next-day price.</p>"
        )
    else:
        r = joined["avg_sentiment"].corr(joined["next_day_return"])
        scatter = go.Figure(
            go.Scatter(
                x=joined["avg_sentiment"], y=joined["next_day_return"], mode="markers",
                marker=dict(size=11, color=UP, line=dict(width=2, color="#0b0f2a")),
                customdata=joined["date"].dt.strftime("%b %d"),
                hovertemplate="%{customdata}<br>Sentiment %{x:+.2f}<br>Next-day return %{y:+.2%}<extra></extra>",
            )
        )
        scatter.update_xaxes(title="Average sentiment that day", tickformat="+.2f", showspikes=False)
        scatter.update_yaxes(title="Price change the next day", tickformat="+.1%")
        c1, c2 = st.columns([3, 1])
        with c1:
            show(style(scatter, 360).update_layout(hovermode="closest"))
        with c2:
            render(
                f'<div class="stat"><div class="stat-label">Correlation (Pearson r)</div>'
                f'<div class="stat-value">{r:+.2f}</div></div>'
                f'<div class="stat"><div class="stat-label">Days compared</div>'
                f'<div class="stat-value">{len(joined)}</div></div>'
                '<p class="chart-note">-1 to +1; near 0 means no linear relationship. Correlation is not '
                "causation, and with few days the number is noisy. Treat this as an exploratory signal, "
                "not a trading strategy.</p>"
            )

# ---------- Headlines ----------
with st.container(key="card_headlines"):
    render(
        '<div class="card-title" id="headlines">Headlines driving the mood</div>'
        f'<p class="card-deck">The most positive and most negative {coin_name} items in this period. '
        "Click one to read it.</p>"
    )
    positive_col, negative_col = st.columns(2, gap="large")
    with positive_col:
        render(headline_list(coin_posts.nlargest(5, "sentiment"), "Most positive", UP))
    with negative_col:
        render(headline_list(coin_posts.nsmallest(5, "sentiment"), "Most negative", DOWN))

headline_cols = {
    "sentiment": st.column_config.NumberColumn("Score", format="%+.2f", width="small"),
    "title": st.column_config.TextColumn("Headline / post", width="large"),
    "source": st.column_config.TextColumn("Source", width="small"),
    "url": st.column_config.LinkColumn("Link", display_text="Open", width="small"),
}
with st.expander(f"All {len(coin_posts):,} {coin_name} items in this period"):
    st.dataframe(
        coin_posts.sort_values("published_at", ascending=False)[["published_at", *headline_cols, "label"]],
        column_config={
            **headline_cols,
            "published_at": st.column_config.DatetimeColumn("Published (UTC)", format="MMM D, HH:mm"),
            "label": st.column_config.TextColumn("Label", width="small"),
        },
        hide_index=True, width="stretch", height=400,
    )

with st.expander("Daily data table"):
    table = coin_daily.merge(
        prices[prices["coin"] == coin][["date", "close", "next_day_return"]], on="date", how="left"
    ).sort_values("date", ascending=False)
    st.dataframe(
        table.drop(columns="coin"),
        column_config={
            "date": st.column_config.DateColumn("Date"),
            "mentions": st.column_config.NumberColumn("Mentions"),
            "avg_sentiment": st.column_config.NumberColumn("Avg. sentiment", format="%+.3f"),
            "positive": st.column_config.NumberColumn("Positive"),
            "neutral": st.column_config.NumberColumn("Neutral"),
            "negative": st.column_config.NumberColumn("Negative"),
            "close": st.column_config.NumberColumn("Close (USD)", format="dollar"),
            "next_day_return": st.column_config.NumberColumn("Next-day return", format="percent"),
        },
        hide_index=True, width="stretch",
    )

with st.expander("How this works"):
    st.markdown(
        f"""
1. **Collect.** Every 6 hours a GitHub Actions job pulls headlines from CoinDesk, Cointelegraph, Decrypt and
   Google News, plus new posts from r/Bitcoin, r/ethereum, r/solana and r/CryptoCurrency (public RSS feeds,
   no API keys). Daily open, high, low and close prices come from Yahoo Finance.
2. **Tag.** Items from general sources are assigned to a coin when they mention it by name or ticker.
   The same headline from two outlets is stored once.
3. **Score.** Each headline (plus the first 500 characters of the summary or post) is scored with
   [VADER](https://github.com/cjhutto/vaderSentiment), extended with crypto slang such as *bullish*, *rekt*
   and *FUD*. Scores ≥ {POSITIVE_THRESHOLD} count as positive and ≤ {NEGATIVE_THRESHOLD} as negative.
4. **Compare.** Daily average sentiment is lined up with the next day's price change.

**Limitations.** VADER is a word-based model, so it misses sarcasm and context. Mention counts reflect what
the feeds return (Google News caps each query at 100 results), not total market chatter. Reddit sometimes
rate-limits automated readers, so some runs may contain news only.
"""
    )

failed = [name for name, result in source_status.items() if result.startswith("failed")]
render(
    '<div class="footer">'
    + (f"Sources unavailable in the latest run: {', '.join(failed)}<br>" if failed else "")
    + f'Built by Ahmad Ammar · <a href="{REPO_URL}" target="_blank" rel="noopener">Source code</a> · '
    '<a href="https://z3us-23.github.io/Ahmad-Ammar.github.io/" target="_blank" rel="noopener">Portfolio</a> · '
    '<a href="https://github.com/Z3US-23" target="_blank" rel="noopener">GitHub</a></div>'
)
