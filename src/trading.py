from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


SUPPORTED_LEVERAGES = {1, 10, 50, 100, 200}


@dataclass
class Position:
    side: str
    entry: float
    size: float
    stop_loss: float
    take_profit: float
    leverage: int
    fee_paid: float = 0.0

    def position_value(self) -> float:
        return self.entry * self.size

    def margin_required(self) -> float:
        return self.position_value() / self.leverage

    def unrealized_pnl(self, price: float) -> float:
        if self.side == "long":
            return (price - self.entry) * self.size
        return (self.entry - price) * self.size


@dataclass
class Account:
    balance: float
    liquidation_ratio: float = 0.5
    positions: List[Position] = field(default_factory=list)

    def equity(self, price: float) -> float:
        return self.balance + sum(p.unrealized_pnl(price) for p in self.positions)

    def margin_used(self) -> float:
        return sum(p.margin_required() for p in self.positions)

    def free_margin(self, price: float) -> float:
        return self.equity(price) - self.margin_used()

    def can_open(self, position: Position, current_price: float) -> bool:
        return self.free_margin(current_price) >= position.margin_required()

    def open_position(self, position: Position, current_price: float) -> bool:
        if position.leverage not in SUPPORTED_LEVERAGES:
            raise ValueError("Levier non supporté")
        if self.can_open(position, current_price):
            self.positions.append(position)
            return True
        return False

    def close_position(self, idx: int, exit_price: float, fee: float = 0.0) -> float:
        pos = self.positions.pop(idx)
        pnl = pos.unrealized_pnl(exit_price) - fee - pos.fee_paid
        self.balance += pnl
        return pnl

    def enforce_liquidation(self, price: float) -> List[float]:
        closed = []
        threshold = self.margin_used() * self.liquidation_ratio
        if self.margin_used() > 0 and self.equity(price) < threshold:
            while self.positions:
                closed.append(self.close_position(0, price, fee=0.0))
        return closed


def compute_spread(volatility: float, liquidity: float, base_spread: float = 0.15) -> float:
    return base_spread * (1.0 + 2.0 * volatility) * (1.0 + 1.0 / max(liquidity, 1e-6))


def compute_slippage(order_size: float, liquidity: float, volatility: float) -> float:
    return (order_size / max(liquidity, 1e-6)) * (1.0 + volatility)


def compute_fee(position_value: float, fee_rate: float = 0.0002) -> float:
    return position_value * fee_rate
