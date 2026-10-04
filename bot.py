"""
Gold news bot for Telegram.

Sources (both from Fair Economy, the company behind ForexFactory):
  - Calendar: MetalsMine weekly calendar JSON feed (metals-specific impact ratings)
  - News:     https://www.metalsmine.com/news (scraped HTML)

Translation and gold-relevance filtering: Google Gemini API.
All times are shown in Europe/Tallinn time (DST handled automatically).

Designed to run every 30 minutes (e.g. GitHub Actions). State is kept in state.json.
"""

import html
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TALLINN = ZoneInfo("Europe/Tallinn")
BASE_URL = "https://www.metalsmine.com"
NEWS_URL = BASE_URL + "/news"
CALENDAR_URL = "https://nfs.faireconomy.media/mm_calendar_thisweek.json"
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = (os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash").strip()
DRY_RUN = os.environ.get("DRY_RUN") == "1"

EVENT_IMPACTS = {"High", "Medium"}          # calendar impact levels to send
NEWS_IMPORTANCE = {"high", "medium"}         # news importance levels to send
DAILY_BRIEF_HOUR = 8                         # Tallinn hour for the daily calendar brief
ALERT_LEAD_MINUTES = 45                      # alert window before an event
ALERT_LATE_MINUTES = 10                      # still alert if the event started this recently
MAX_NEWS_AGE_HOURS = 12                      # ignore older news
MAX_NEWS_PER_RUN = 8                         # news items sent to Gemini per run
FAILURE_WARN_THRESHOLD = 8                   # consecutive failures before a Telegram warning

# Calendar events that are about base metals, not gold.
EXCLUDE_EVENT_WORDS = ("copper", "aluminum", "aluminium", "nickel", "zinc", "iron ore", "lme ")

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

NEWS_HREF = re.compile(r"^(?:https://www\.metalsmine\.com)?/news/(\d+)-[A-Za-z0-9-]+$")
IMPACT_SRC = re.compile(r"/impact/[a-z]+/(high|medium|low)\.svg")

FLAGS = {
    "USD": "🇺🇸", "EUR": "🇪🇺", "GBP": "🇬🇧", "JPY": "🇯🇵", "CNY": "🇨🇳",
    "CHF": "🇨🇭", "CAD": "🇨🇦", "AUD": "🇦🇺", "NZD": "🇳🇿", "INR": "🇮🇳",
    "RUB": "🇷🇺", "ZAR": "🇿🇦", "MXN": "🇲🇽", "All": "🌐",
}
FA_WEEKDAYS = ["دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه", "شنبه", "یکشنبه"]
FA_MONTHS = ["ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن", "ژوئیه",
             "اوت", "سپتامبر", "اکتبر", "نوامبر", "دسامبر"]
IMPACT_ICON = {"high": "🟥", "medium": "🟧", "low": "🟨"}
IMPACT_FA = {"high": "بالا", "medium": "متوسط", "low": "پایین"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def esc(text):
    return html.escape(str(text or ""), quote=False)


def fa_date(dt):
    d = dt.astimezone(TALLINN)
    return f"{FA_WEEKDAYS[d.weekday()]} {d.day} {FA_MONTHS[d.month - 1]}"


def fa_time(dt):
    return dt.astimezone(TALLINN).strftime("%H:%M")


def clip(text, limit):
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    state.setdefault("initialized", False)
    state.setdefault("seen_news", [])
    state.setdefault("alerted_events", [])
    state.setdefault("last_brief", "")
    state.setdefault("failures", {"news": 0, "calendar": 0})
    state.setdefault("warned", {"news": False, "calendar": False})
    return state


def save_state(state):
    state["seen_news"] = state["seen_news"][-600:]
    state["alerted_events"] = state["alerted_events"][-300:]
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send(text):
    if DRY_RUN:
        print(text)
        print("-" * 60)
        return True
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    for _ in range(3):
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code == 429:
            wait = r.json().get("parameters", {}).get("retry_after", 5)
            time.sleep(int(wait) + 1)
            continue
        if not r.ok:
            print(f"Telegram error {r.status_code}: {r.text}", file=sys.stderr)
            return False
        time.sleep(1.2)
        return True
    return False


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def gemini_json(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
    }
    headers = {"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"}
    last_error = None
    for attempt in range(3):
        r = requests.post(url, headers=headers, json=body, timeout=120)
        if r.status_code in (429, 500, 502, 503):
            last_error = f"{r.status_code}: {r.text[:300]}"
            time.sleep(20 * (attempt + 1))
            continue
        if not r.ok:
            raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:500]}")
        data = r.json()
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        return json.loads(text)
    raise RuntimeError(f"Gemini unavailable after retries ({last_error})")


EVENT_PROMPT = """You are a financial translator for a Persian-language Telegram channel about GOLD (XAU/USD).
For each economic calendar event below, return a JSON array of objects:
  {"i": <same index>, "title_fa": "<natural Persian name of the event>",
   "why_fa": "<one short Persian sentence: why this event can move the gold price and in which direction a stronger/weaker result usually pushes gold>"}
Rules: write fluent Persian; keep standard abbreviations (CPI, FOMC, NFP) in Latin letters inside the Persian text;
do not give trading advice; do not invent numbers. Return ONLY the JSON array.
Events:
"""

NEWS_PROMPT = """You are an editor for a Persian-language Telegram channel that covers ONLY news relevant to GOLD (XAU/USD).
Below are news items scraped from a metals news site. Each has an id, a title and raw text
(raw text contains the title, source, age, and an excerpt of the article).
For each item return a JSON object:
  {"id": "<same id>",
   "gold_relevant": true/false,
   "importance": "high" | "medium" | "low",
   "title_fa": "<Persian translation of the title>",
   "summary_fa": "<2-3 sentence Persian summary of the excerpt>",
   "why_gold_fa": "<one Persian sentence on the likely effect on gold; say clearly if the effect is uncertain>",
   "translation_fa": "<complete, faithful Persian translation of the excerpt text only (not the source/age/comment counts); do not add anything that is not in the text>"}
gold_relevant = true ONLY if the item is about gold itself, or is very likely to move the gold price:
  Fed / US interest rates, US inflation or jobs data, the US dollar broadly, US Treasury yields,
  major geopolitical or war risk, central-bank gold buying/selling, gold ETFs, large gold miners.
gold_relevant = false for: copper/silver/platinum-only news, cryptocurrencies, technical analysis of
specific currency pairs, entertainment, celebrity or crime stories, generic politics without a market angle.
importance: how much this could move gold today (high = major market mover, medium = notable, low = minor).
Write fluent Persian. Keep tickers and abbreviations (XAU/USD, CPI, FOMC, ETF) in Latin letters.
Do not give trading advice. Return ONLY the JSON array.
Items:
"""


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

def fetch_calendar():
    r = requests.get(CALENDAR_URL, headers=HTTP_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def gold_events(raw_events):
    events = []
    for e in raw_events:
        if e.get("impact") not in EVENT_IMPACTS:
            continue
        title = e.get("title", "")
        if any(w in (title.lower() + " ") for w in EXCLUDE_EVENT_WORDS):
            continue
        try:
            dt = datetime.fromisoformat(e["date"])
        except (KeyError, ValueError):
            continue
        item = dict(e)
        item["dt"] = dt
        item["key"] = f'{e["date"]}|{e.get("country", "")}|{title}'
        events.append(item)
    events.sort(key=lambda x: x["dt"])
    return events


def explain_events(events):
    if not events:
        return {}
    items = [
        {"i": i, "title": e["title"], "country": e.get("country", ""),
         "impact": e["impact"], "forecast": e.get("forecast", ""), "previous": e.get("previous", "")}
        for i, e in enumerate(events)
    ]
    try:
        result = gemini_json(EVENT_PROMPT + json.dumps(items, ensure_ascii=False))
        return {int(x["i"]): x for x in result if isinstance(x, dict) and "i" in x}
    except Exception as exc:  # fall back to English titles
        print(f"Event translation failed: {exc}", file=sys.stderr)
        return {}


def event_lines(e, info):
    impact = e["impact"].lower()
    flag = FLAGS.get(e.get("country", ""), "")
    title_fa = info.get("title_fa") or e["title"]
    lines = [f'{IMPACT_ICON.get(impact, "")} <b>{esc(fa_time(e["dt"]))}</b> | {flag} {esc(e.get("country", ""))} | '
             f'<b>{esc(title_fa)}</b>']
    lines.append(f'   <i>{esc(e["title"])}</i> — اهمیت: {IMPACT_FA.get(impact, impact)}')
    nums = []
    if e.get("forecast"):
        nums.append(f'پیش‌بینی: {esc(e["forecast"])}')
    if e.get("previous"):
        nums.append(f'قبلی: {esc(e["previous"])}')
    if nums:
        lines.append("   " + " | ".join(nums))
    if info.get("why_fa"):
        lines.append(f'   💡 {esc(info["why_fa"])}')
    return "\n".join(lines)


def run_calendar(state, now_utc):
    events = gold_events(fetch_calendar())
    now_tll = now_utc.astimezone(TALLINN)
    today = now_tll.date().isoformat()

    # Daily brief
    if now_tll.hour >= DAILY_BRIEF_HOUR and state["last_brief"] != today:
        todays = [e for e in events if e["dt"].astimezone(TALLINN).date() == now_tll.date()]
        header = f"📅 <b>تقویم اقتصادی امروز برای طلا</b>\n{fa_date(now_tll)} — ساعت‌ها به وقت تالین\n"
        if todays:
            info = explain_events(todays)
            body = "\n\n".join(event_lines(e, info.get(i, {})) for i, e in enumerate(todays))
        else:
            body = "امروز ایونت با اهمیت بالا یا متوسط برای طلا در تقویم نیست."
        if send(header + "\n" + body):
            state["last_brief"] = today

    # Alerts shortly before each event
    upcoming = []
    for e in events:
        minutes = (e["dt"] - now_utc).total_seconds() / 60
        if -ALERT_LATE_MINUTES <= minutes <= ALERT_LEAD_MINUTES and e["key"] not in state["alerted_events"]:
            upcoming.append((e, minutes))
    if upcoming:
        info = explain_events([e for e, _ in upcoming])
        for i, (e, minutes) in enumerate(upcoming):
            when = f"حدود {int(round(minutes))} دقیقه دیگر" if minutes > 1 else "همین الان"
            text = (f"⏰ <b>هشدار ایونت طلا</b> — {when}\n"
                    f"🕒 {fa_date(e['dt'])}، ساعت {fa_time(e['dt'])} به وقت تالین\n\n"
                    + event_lines(e, info.get(i, {})))
            if send(text):
                state["alerted_events"].append(e["key"])


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def item_container(link):
    """Largest ancestor of the link that contains links to only one news item."""
    best = link
    node = link
    while node.parent is not None and node.parent.name not in ("body", "html", "[document]"):
        parent = node.parent
        ids = set()
        for a in parent.find_all("a", href=True):
            m = NEWS_HREF.match(a["href"].strip())
            if m:
                ids.add(m.group(1))
        if len(ids) > 1:
            break
        best = parent
        node = parent
    return best


def parse_age(text, now_utc):
    """Return (published_datetime_utc, is_precise) or (None, False)."""
    m = re.search(r"\|\s*((?:\d+\s*(?:day|hr|min|sec)s?\s*)+ago)", text)
    if m:
        s = m.group(1)
        delta = timedelta()
        for num, unit in re.findall(r"(\d+)\s*(day|hr|min|sec)", s):
            n = int(num)
            delta += {"day": timedelta(days=n), "hr": timedelta(hours=n),
                      "min": timedelta(minutes=n), "sec": timedelta(seconds=n)}[unit]
        return now_utc - delta, True
    m = re.search(r"\|\s*([A-Z][a-z]{2} \d{1,2}, \d{4})", text)
    if m:
        try:
            d = datetime.strptime(m.group(1), "%b %d, %Y").replace(tzinfo=timezone.utc)
            return d, False
        except ValueError:
            pass
    return None, False


def parse_news(page_html, now_utc):
    soup = BeautifulSoup(page_html, "html.parser")
    items, order = {}, []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        m = NEWS_HREF.match(href)
        if not m:
            continue
        title = a.get_text(" ", strip=True)
        nid = m.group(1)
        if len(title) < 8 or nid in items:
            continue
        box = item_container(a)
        raw = box.get_text(" ", strip=True)
        impact = None
        for img in box.find_all("img", src=True):
            im = IMPACT_SRC.search(img["src"])
            if im:
                impact = im.group(1)
                break
        src = re.search(r"From\s+(\S+)", raw)
        published, precise = parse_age(raw, now_utc)
        items[nid] = {
            "id": nid,
            "title": title,
            "url": href if href.startswith("http") else BASE_URL + href,
            "raw": clip(raw, 1800),
            "site_impact": impact,
            "source": src.group(1) if src else "",
            "published": published,
            "precise": precise,
        }
        order.append(nid)
    return [items[i] for i in order]


def fetch_news(now_utc):
    r = requests.get(NEWS_URL, headers=HTTP_HEADERS, timeout=30)
    r.raise_for_status()
    items = parse_news(r.text, now_utc)
    if not items:
        raise RuntimeError("News page loaded but no news items were found (page layout may have changed).")
    return items


def news_message(item, ai):
    if item["site_impact"]:
        level, origin = item["site_impact"], "رتبه‌بندی خود سایت"
    else:
        level, origin = ai.get("importance", "low"), "تخمین هوش مصنوعی"
    if item["published"] and item["precise"]:
        when = f"🕒 {fa_date(item['published'])}، ساعت {fa_time(item['published'])} به وقت تالین"
    elif item["published"]:
        when = f"🕒 {fa_date(item['published'])}"
    else:
        when = f"🕒 دریافت‌شده: {fa_time(datetime.now(timezone.utc))} به وقت تالین"
    parts = [
        f"{IMPACT_ICON.get(level, '')} <b>{esc(ai.get('title_fa') or item['title'])}</b>",
        when,
        f"📊 اهمیت: {IMPACT_FA.get(level, level)} ({origin})",
        "",
        f"📝 <b>خلاصه:</b> {esc(clip(ai.get('summary_fa'), 700))}",
        "",
        f"🟡 <b>اثر روی طلا:</b> {esc(clip(ai.get('why_gold_fa'), 400))}",
        "",
        f"📄 <b>ترجمه‌ی متن منتشرشده:</b>\n{esc(clip(ai.get('translation_fa'), 2200))}",
        "",
        f'🔗 <a href="{esc(item["url"])}">منبع: {esc(item["source"] or "MetalsMine")}</a>',
    ]
    return "\n".join(parts)


def run_news(state, now_utc):
    items = fetch_news(now_utc)
    seen = set(state["seen_news"])

    first_run = not state["initialized"]
    state["initialized"] = True
    # First run: backfill only today's news (Tallinn date); later runs use the normal age window.
    if first_run:
        today_start = now_utc.astimezone(TALLINN).replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = today_start.astimezone(timezone.utc)
    else:
        cutoff = now_utc - timedelta(hours=MAX_NEWS_AGE_HOURS)
    fresh = []
    for it in items:
        if it["id"] in seen:
            continue
        if it["published"] is None or it["published"] < cutoff or not it["precise"]:
            state["seen_news"].append(it["id"])   # too old or undated: skip permanently
            continue
        fresh.append(it)

    fresh = fresh[:MAX_NEWS_PER_RUN]
    if not fresh:
        return

    payload = [{"id": it["id"], "title": it["title"], "raw": it["raw"]} for it in fresh]
    result = gemini_json(NEWS_PROMPT + json.dumps(payload, ensure_ascii=False))
    by_id = {str(x.get("id")): x for x in result if isinstance(x, dict)}

    # Send oldest first so the channel reads chronologically.
    for it in sorted(fresh, key=lambda x: x["published"]):
        ai = by_id.get(it["id"])
        if ai is None:
            continue  # model skipped it; retry next run
        level = it["site_impact"] or str(ai.get("importance", "low")).lower()
        if ai.get("gold_relevant") and level in NEWS_IMPORTANCE:
            if not send(news_message(it, ai)):
                continue
        state["seen_news"].append(it["id"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def track(state, name, ok, error=None):
    labels = {"news": "صفحه‌ی خبر MetalsMine", "calendar": "تقویم اقتصادی"}
    if ok:
        state["failures"][name] = 0
        state["warned"][name] = False
        return
    state["failures"][name] += 1
    print(f"[{name}] failure #{state['failures'][name]}: {error}", file=sys.stderr)
    if state["failures"][name] >= FAILURE_WARN_THRESHOLD and not state["warned"][name]:
        if send(f"⚠️ ربات چند بار پشت سر هم نتوانست {labels[name]} را بخواند.\n"
                f"خطا: <code>{esc(clip(error, 300))}</code>\n"
                "جزئیات در بخش Actions گیت‌هاب."):
            state["warned"][name] = True


def main():
    required = {"GEMINI_API_KEY": GEMINI_KEY}
    if not DRY_RUN:
        required.update({"TELEGRAM_BOT_TOKEN": BOT_TOKEN, "TELEGRAM_CHAT_ID": CHAT_ID})
    missing = [k for k, v in required.items() if not v]
    if missing:
        print("Missing environment variables: " + ", ".join(missing), file=sys.stderr)
        sys.exit(1)

    state = load_state()
    now_utc = datetime.now(timezone.utc)

    if not state.get("welcomed"):
        state["welcomed"] = send("✅ <b>ربات خبر طلا فعال شد</b>\n"
             "از این به بعد خبرها و ایونت‌های مهم و متوسط مربوط به طلا، به فارسی و به وقت تالین، اینجا ارسال می‌شود.\n"
             f"خلاصه‌ی روزانه‌ی تقویم: هر روز حدود ساعت {DAILY_BRIEF_HOUR}:00 صبح.")

    try:
        run_calendar(state, now_utc)
        track(state, "calendar", True)
    except Exception as exc:
        track(state, "calendar", False, repr(exc))

    try:
        run_news(state, now_utc)
        track(state, "news", True)
    except Exception as exc:
        track(state, "news", False, repr(exc))

    save_state(state)


if __name__ == "__main__":
    main()
