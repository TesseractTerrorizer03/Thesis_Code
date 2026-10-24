"""Download optional sector participation and SPY volume inputs into Thesis/data.

Run with the project's existing Python environment. Analysis notebooks only read
the resulting cache; they never download data or change the historical cache.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
SECTORS = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']


def main():
    yf.set_tz_cache_location(str(ROOT / '.cache/yfinance'))
    raw = yf.download(SECTORS+['SPY'], start='2006-09-28', end='2026-09-02',
                      auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        raise RuntimeError('Historical download returned no data.')
    closes = raw['Close'].reindex(columns=SECTORS)
    volume = raw['Volume'][['SPY']].rename(columns={'SPY': 'SPY_volume'})
    data = closes.join(volume)
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    data = data.sort_index()
    if not data.index.is_unique or data.isna().any().any() or (data <= 0).any().any():
        raise RuntimeError('Download is incomplete or contains invalid prices/volume; cache was not written.')
    target = ROOT / 'data/sector_prices_and_volume.csv'
    if target.exists():
        raise FileExistsError('Cache exists; move it to an explicitly versioned filename before refreshing.')
    data.to_csv(target, index_label='date')
    metadata = dict(source='Yahoo Finance through yfinance; adjusted sector closes and reported SPY volume',
                    fetched_at_utc=datetime.now(timezone.utc).isoformat(),
                    start='2006-09-28', end_exclusive='2026-09-02', sectors=SECTORS,
                    rows=len(data), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    note='Fixed sector-ETF panel, not constituent-level stock breadth. Current-vintage adjusted prices; historical sector definitions change.')
    (ROOT / 'data/sector_prices_and_volume_provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved {len(data):,} rows: {target}')


if __name__ == '__main__':
    main()
