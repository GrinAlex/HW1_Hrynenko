from datetime import date
import json
import re
from typing import Literal
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Початкова in-memory база цінових акцій
PROMO_DB: dict[str, dict] = {
    "TECH_START": {
        "promo_code": "TECH_START",
        "category": "electronics",
        "discount_percent": 15,
        "start_date": "2026-11-01",
        "end_date": "2026-12-31",
        "active": True,
    },
    "HOME_COZY": {
        "promo_code": "HOME_COZY",
        "category": "home",
        "discount_percent": 20,
        "start_date": "2026-11-10",
        "end_date": "2026-12-20",
        "active": True,
    },
}


# ── Pydantic v2 схеми параметрів ───────────────────────────────────────────

class SearchPromosInput(BaseModel):
    """Параметри пошуку цінових акцій."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: Literal["electronics", "clothing", "home", "all"] = Field(
        default="all",
        description="Категорія товарів для фільтрації: 'electronics', 'clothing', 'home' або 'all'.",
    )
    only_active: bool = Field(
        default=True,
        description="Якщо True — повертає лише активні акції, якщо False — усі.",
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        allowed = {"electronics", "clothing", "home", "all"}
        if v not in allowed:
            raise ValueError(f"Непідтримувана категорія '{v}'. Доступні: {sorted(allowed)}")
        return v


class CreatePromoInput(BaseModel):
    """Параметри створення нової цінової акції з бізнес-обмеженнями."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    promo_code: str = Field(
        description="Унікальний промокод у верхньому регістрі (4–20 символів, лише A-Z, 0-9 та _), напр. BLACK_FRIDAY_26."
    )
    category: Literal["electronics", "clothing", "home", "all"] = Field(
        description="Категорія товарів, на яку діє акція ('electronics', 'clothing', 'home' або 'all')."
    )
    discount_percent: int = Field(
        description="Розмір знижки у відсотках (від 1 до 70%). Знижки понад 70% заборонені."
    )
    start_date: date = Field(description="Дата початку акції у форматі YYYY-MM-DD.")
    end_date: date = Field(description="Дата завершення акції у форматі YYYY-MM-DD.")

    @field_validator("promo_code")
    @classmethod
    def validate_promo_code(cls, v: str) -> str:
        if not (4 <= len(v) <= 20) or not re.match(r"^[A-Z0-9_]+$", v):
            raise ValueError(
                "Промокод має містити 4–20 символів у верхньому регістрі (лише A-Z, 0-9 та _)."
            )
        return v

    @field_validator("discount_percent")
    @classmethod
    def validate_discount(cls, v: int) -> int:
        if not 1 <= v <= 70:
            raise ValueError("Знижка має бути в діапазоні від 1 до 70%.")
        return v

    @model_validator(mode="after")
    def validate_dates_order(self) -> "CreatePromoInput":
        if self.end_date <= self.start_date:
            raise ValueError("Дата завершення (end_date) має бути пізнішою за дату початку (start_date).")
        return self


class DeactivatePromoInput(BaseModel):
    """Параметри дострокової деактивації акції."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    promo_code: str = Field(description="Промокод діючої акції для деактивації.")
    reason: str = Field(description="Причина зупинки акції (мінімум 5 символів).")

    @field_validator("promo_code")
    @classmethod
    def validate_promo_code(cls, v: str) -> str:
        if not (4 <= len(v) <= 20) or not re.match(r"^[A-Z0-9_]+$", v):
            raise ValueError("Некоректний формат промокоду (очікуються A-Z, 0-9, _).")
        return v

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError("Причина деактивації має містити мінімум 5 символів.")
        return v.strip()


# ── Інструменти (Tools) ─────────────────────────────────────────────────────

@tool(args_schema=SearchPromosInput)
def search_promos(category: str = "all", only_active: bool = True) -> str:
    """Шукає наявні цінові акції в базі даних за категорією товарів та статусом.

    Використовуйте цей інструмент, коли потрібно переглянути список акцій
    або перевірити наявні пропозиції перед створенням нової акції.
    """
    results = []
    for promo in PROMO_DB.values():
        if category != "all" and promo["category"] != category:
            continue
        if only_active and not promo["active"]:
            continue
        results.append(promo)
    return json.dumps({"status": "success", "count": len(results), "promos": results}, ensure_ascii=False)


@tool(args_schema=CreatePromoInput)
def create_promo(
    promo_code: str,
    category: str,
    discount_percent: int,
    start_date: date,
    end_date: date,
) -> str:
    """Створює нову цінову акцію або промокод у системі.

    Використовуйте цей інструмент для запуску нової акції.
    Знижка не може перевищувати 70%, а промокод має бути унікальним.
    """
    if promo_code in PROMO_DB:
        return json.dumps(
            {"status": "error", "error_type": "business_rule", "message": f"Промокод '{promo_code}' вже існує."},
            ensure_ascii=False,
        )

    new_promo = {
        "promo_code": promo_code,
        "category": category,
        "discount_percent": discount_percent,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "active": True,
    }
    PROMO_DB[promo_code] = new_promo
    return json.dumps({"status": "success", "promo": new_promo}, ensure_ascii=False)


@tool(args_schema=DeactivatePromoInput)
def deactivate_promo(promo_code: str, reason: str) -> str:
    """Достроково деактивує діючу акцію за її промокодом.

    Використовуйте цей інструмент, коли потрібно зупинити активну акцію
    з обов'язковим зазначенням причини.
    """
    promo = PROMO_DB.get(promo_code)
    if not promo:
        return json.dumps(
            {"status": "error", "error_type": "not_found", "message": f"Промокод '{promo_code}' не знайдено."},
            ensure_ascii=False,
        )
    if not promo["active"]:
        return json.dumps(
            {"status": "error", "error_type": "business_rule", "message": f"Акція '{promo_code}' вже неактивна."},
            ensure_ascii=False,
        )

    promo["active"] = False
    promo["deactivation_reason"] = reason
    return json.dumps({"status": "success", "promo": promo}, ensure_ascii=False)


TOOLS_LIST = [search_promos, create_promo, deactivate_promo]