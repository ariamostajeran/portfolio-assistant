# Portfolio Website

## Summary
A production Flask web application that doubles as a personal CV landing page and a live ML engineering showcase. Built to demonstrate end-to-end web development and MLOps skills to employers while finishing an MSc at TU/e. The site combines a single-page portfolio (experience, education, skills, awards) with two dedicated interactive project pages: a live stock trading signal dashboard powered by Databricks, and a full FIFA World Cup 2022 football scouting tool with five interactive Plotly.js charts. The problem it solves is standing out — rather than a static PDF CV, every project is demonstrable in the browser.

## Tech stack
- **Flask 3.0.3** — lightweight Python web framework; chosen for simplicity and fast iteration, no ORM or heavy framework needed for a portfolio
- **Jinja2** — Flask's built-in templating; block inheritance (base.html) keeps all pages consistent with one navbar/footer
- **Chart.js 4.4.0** — used on the stock MLOps page for cumulative returns and feature importance bar charts; CDN-loaded, zero build step
- **Plotly.js 2.27.0** — used on the ScoutBall page for five chart types (scatter/swarm, parallel coordinates, radar, choropleth); chosen because the original Dash app was also Plotly under the hood
- **Gunicorn** — WSGI server, 2 workers; sits in front of Flask in production
- **Docker + Docker Compose** — containerised with Python 3.11-slim; Compose used for local dev with env var injection
- **Render.com free tier** — PaaS deployment; auto-deploys from GitHub main branch on push
- **python-dotenv** — environment variable management; .env never committed, .env.example checked in
- **Inter + JetBrains Mono** — Google Fonts; Inter for body/UI, JetBrains Mono for numbers and metric values

## What it does
1. **Landing page (`/`)** — Single-page CV with sections: Hero (name, bio, links to GitHub/LinkedIn/Scholar), About, Experience (7 timeline entries), Education (MSc TU/e + BSc AUT), Skills (tagged grid by category), Projects (featured card + grid of 6), Awards & certifications.
2. **Stock MLOps page (`/projects/stock-mlops`)** — Calls `data_fetcher.py` on page load. If `DATABRICKS_HOST` and `DATABRICKS_TOKEN` env vars are set, fetches `latest_signals.json` from Databricks DBFS via REST API (`/api/2.0/dbfs/read`), base64-decodes the response, and parses JSON. Result is cached in-memory for 6 hours. If credentials are absent or the request fails, falls back to hardcoded mock data. The template renders signal cards (LONG/SHORT/NO_TRADE with probability bars), backtest metrics grid (hit rate, Sharpe, cumulative return, trades), two Chart.js charts (cumulative returns by month, feature importance), and a pipeline diagram.
3. **ScoutBall page (`/projects/scoutball`)** — Serves a static page; all data is loaded client-side from `/static/data/scoutball_data.json` (209 KB, 680 players). JavaScript (`scoutball.js`) manages all state: position tabs (FW/MF/DF/GK), age range slider, six position-specific stat sliders, a player name search bar with live dropdown, and click-to-select on the swarm plot. Five Plotly.js charts update reactively: swarm/scatter plot (overall score distribution), parallel coordinates (multi-stat view, shows selected players' lines in their radar colours), radar chart (normalised stats for up to 3 players), choropleth world map (player counts by country), and a player detail card. Players can be added to the radar comparison via swarm plot clicks, parallel coordinates hover+click, or the search bar.

## Key technical decisions
- **Static JSON for ScoutBall data** — The FIFA WC 2022 dataset (12 CSVs, 680 players) is pre-processed once by a Python script into a single `scoutball_data.json` and served as a static file. This avoids a database, keeps the Render free tier viable, and means zero server load per chart interaction — all filtering and rendering is client-side JavaScript.
- **Single parcoords trace with colorscale** — Plotly's `parcoords` trace type does not reliably share axes across multiple traces. When selected players are shown, all their data is packed into one trace with `line.color = [0, 1, 2]` mapped to a discrete colorscale (`[[0, cyan], [0.5, amber], [1, violet]]`). This guarantees all lines appear on the same axes.
- **Event listener attached once on boot** — The swarm plot's `plotly_click` listener is registered once after the initial data fetch, not inside `renderSwarm()`. Registering inside the render function caused duplicate listeners on every re-render, so clicking a second player would call `toggleRadarPlayer` twice (add then immediately remove), leaving the radar empty.
- **Position-fixed axis ranges in parallel coordinates** — Axis ranges are computed from all players of the current position, not just the filtered subset. This keeps axes stable as sliders are moved, so lines don't jump around.
- **6-hour in-memory cache** — Databricks API calls are expensive and the signals only update daily. A simple Python dict (`_cache`, `_cache_ts`) avoids hitting the API on every page load with no external dependency (no Redis needed on the free tier).
- **CSS custom properties throughout** — All colours, radii, and spacing defined as CSS variables in `:root`. Dark navy `#0a0e1a`, cyan `#06b6d4`, card background `#1a2236`. Theming any element is one variable change.
- **Flask app factory pattern** — `create_app()` in `__init__.py` registers the Blueprint; supports multiple environments (dev/prod) without code changes, just env vars.

## What I learned
- **Plotly.js parcoords is opinionated** — It took several iterations to get multi-player parallel coordinates right. The trace/colorscale approach, axis range stability, and hover event handling (parcoords doesn't fire `plotly_click` on lines — needed a `plotly_hover` + native DOM `click` combo) were all non-obvious.
- **Free tier cold starts hurt first impressions** — Render's free tier spins down after 15 minutes of inactivity, causing a 30–60 second cold start. For a portfolio this matters. Keeping all ScoutBall computation client-side (no server round-trips per chart) partly mitigates this.
- **Data cleaning is always the real work** — The Kaggle dataset had 12 CSVs with inconsistent player name encoding (UTF-8 vs cp1252), mixed position formats ("FW,MF"), and age strings in "years-days" format. The processing script handles all of this before generating the JSON.
- **Docker Compose for local parity** — Running locally with `docker-compose up` instead of bare `python run.py` caught env var and port issues before they hit Render.

## Results / metrics
- 680 players processed from 12 CSVs into 209 KB JSON, loaded and rendered client-side in under 1 second on a modern browser
- Stock MLOps backtest metrics (mock): AAPL 58% hit rate, 1.23 Sharpe, +34.2% cumulative return over 47 trades; MSFT 61% hit rate, 1.45 Sharpe, +41.0%
- Blood pressure thesis result surfaced on the page: MAE < 7 mmHg on MIMIC-II benchmark data
- Two dedicated project pages with interactive charts; homepage project cards flagged with "↗ Full project" badge

## Links
- GitHub: github.com/ariamostajeran/aria-portfolio
- Live: aria-portfolio.onrender.com
