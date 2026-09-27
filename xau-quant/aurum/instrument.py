"""Contract specifications for the gold instruments aurum trades.

Every cost the backtester charges comes from here, so a strategy that looks
good on paper has already paid a realistic spread, commission and overnight
financing before it reaches a report.

The defaults describe a typical ECN retail XAUUSD CFD. Brokers differ, so
pull your own numbers from the symbol specification in MT5 (right click the
symbol, Specification) and pass them in.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GoldInstrument:
    symbol: str = "XAUUSD"
    # One standard lot is 100 troy ounces on almost every broker.
    contract_size_oz: float = 100.0
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 50.0
    tick_size: float = 0.01
    # Typical raw spread in USD per ounce. Widens hard around NFP, CPI, FOMC
    # and the Asian rollover, which `spread_at` models.
    spread_usd: float = 0.20
    # Slippage per side, USD per ounce, on market orders and stop fills.
    slippage_usd: float = 0.05
    # Round turn commission per standard lot, USD.
    commission_per_lot: float = 7.0
    # Overnight financing per standard lot per night, USD. Gold longs pay
    # because you are effectively borrowing dollars to hold a zero yield
    # asset; with Fed funds above gold lease rates, shorts usually earn a
    # little or pay a little depending on the broker's markup.
    swap_long_per_lot: float = -45.0
    swap_short_per_lot: float = 8.0
    # Most brokers charge three nights on Wednesday to cover the weekend.
    # Swap is charged at the 17:00 New York rollover.
    triple_swap_weekday: int = 2
    leverage: float = 100.0

    def usd_per_point(self, lots: float) -> float:
        """P&L in USD for a $1.00 move in the gold price."""
        return lots * self.contract_size_oz

    def round_lots(self, lots: float) -> float:
        """Round a raw size down to the broker's lot step, within limits."""
        if lots <= 0:
            return 0.0
        steps = int(lots / self.lot_step + 1e-9)
        lots = round(steps * self.lot_step, 8)
        if lots < self.min_lot:
            return 0.0
        return min(lots, self.max_lot)

    def margin_required(self, lots: float, price: float) -> float:
        return abs(lots) * self.contract_size_oz * price / self.leverage

    def spread_at(self, event_risk: bool = False, rollover: bool = False) -> float:
        """Spread for a bar. News and rollover widen it several times over."""
        mult = 1.0
        if event_risk:
            mult = max(mult, 4.0)
        if rollover:
            mult = max(mult, 3.0)
        return self.spread_usd * mult

    def with_overrides(self, **kwargs) -> "GoldInstrument":
        return replace(self, **kwargs)


XAUUSD = GoldInstrument()

# COMEX gold future. Used for research on futures data (GC=F); positions are
# whole contracts and there is no swap, the roll cost is in the curve.
GC_FUTURE = GoldInstrument(
    symbol="GC",
    contract_size_oz=100.0,
    min_lot=1.0,
    lot_step=1.0,
    max_lot=500.0,
    tick_size=0.10,
    spread_usd=0.10,
    slippage_usd=0.10,
    commission_per_lot=5.0,
    swap_long_per_lot=0.0,
    swap_short_per_lot=0.0,
    leverage=20.0,
)

# COMEX micro gold, 10 oz.
MGC_FUTURE = GC_FUTURE.with_overrides(symbol="MGC", contract_size_oz=10.0, commission_per_lot=1.5)
