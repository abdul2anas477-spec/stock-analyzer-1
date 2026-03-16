from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import yfinance as yf
from groq import Groq
import numpy as np
import os

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_KEY_HERE")

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)

def calc_volatility(closes):
    if len(closes) < 10:
        return None
    returns = np.diff(closes) / closes[:-1]
    return round(float(np.std(returns) * np.sqrt(252) * 100), 1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stock/<ticker>")
def get_stock(ticker):
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period="3mo")

        if hist.empty:
            return jsonify({"error": f"No data found for {ticker.upper()}"}), 404

        info = stock.info
        closes = hist["Close"].values.tolist()
        dates  = [d.strftime("%b %d") for d in hist.index]

        rsi        = calc_rsi(np.array(closes))
        volatility = calc_volatility(np.array(closes))
        ma20       = round(float(np.mean(closes[-20:])), 2) if len(closes) >= 20 else None
        ma50       = round(float(np.mean(closes[-50:])), 2) if len(closes) >= 50 else None

        current = closes[-1]
        prev    = closes[-2] if len(closes) > 1 else current
        change_pct = round((current - prev) / prev * 100, 2)

        prices = [{"date": dates[i], "price": round(closes[i], 2)} for i in range(len(closes))]

        return jsonify({
            "symbol":        ticker.upper(),
            "name":          info.get("longName") or info.get("shortName") or ticker.upper(),
            "exchange":      info.get("exchange", ""),
            "currency":      info.get("currency", "USD"),
            "currentPrice":  round(current, 2),
            "previousClose": round(prev, 2),
            "changePct":     change_pct,
            "high52w":       info.get("fiftyTwoWeekHigh"),
            "low52w":        info.get("fiftyTwoWeekLow"),
            "rsi":           rsi,
            "ma20":          ma20,
            "ma50":          ma50,
            "volatility":    volatility,
            "prices":        prices,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        d = request.json
        client = Groq(api_key=GROQ_API_KEY)

        prompt = f"""You are a stock risk analyst. Analyze this data and give a concise risk management report.

Stock: {d['name']} ({d['symbol']})
Price: {d['currency']} {d['currentPrice']} ({'+' if d['changePct']>=0 else ''}{d['changePct']}% today)
RSI-14: {d['rsi']} | MA20: {d['ma20']} | MA50: {d['ma50']} | Volatility: {d['volatility']}% annually
52W High: {d['high52w']} | 52W Low: {d['low52w']}

Use these exact bold headers:
**Signal:** BUY / SELL / HOLD — one sentence reason.
**Risk Level:** Low / Medium / High — brief explanation.
**Support & Resistance:** Key price levels.
**Risk Management Tips:** 3 practical tips (stop-loss, position size, key levels).
**Caveat:** One important limitation.

Max 220 words. No guaranteed predictions."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )
        return jsonify({"analysis": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
