"""Provider ingestion and immutable dataset publication boundary."""

from .coverage import (
    CoverageAudit,
    CoverageGap,
    MemberCountAnomaly,
    SecurityLifecycle,
    UniverseMembership,
    audit_historical_coverage,
)
from .freshness import FreshnessAssessment, assess_dataset_freshness
from .tushare_client import TokenMissingError, TushareClient
from .tushare_models import (
    AdjFactorRecord,
    DailyBarRecord,
    StockBasicRecord,
    TradeCalRecord,
)
from .tushare_transport import (
    URL_ENV_VAR,
    HttpJsonTransport,
    SecretToken,
    Transport,
    TransportRequest,
    TransportResponse,
    TushareAPIError,
    TushareTransientError,
    resolve_base_url,
)

__all__ = [
    "AdjFactorRecord",
    "CoverageAudit",
    "CoverageGap",
    "DailyBarRecord",
    "FreshnessAssessment",
    "HttpJsonTransport",
    "MemberCountAnomaly",
    "SecretToken",
    "SecurityLifecycle",
    "StockBasicRecord",
    "TokenMissingError",
    "TradeCalRecord",
    "Transport",
    "TransportRequest",
    "TransportResponse",
    "TushareAPIError",
    "TushareClient",
    "TushareTransientError",
    "URL_ENV_VAR",
    "UniverseMembership",
    "assess_dataset_freshness",
    "audit_historical_coverage",
    "resolve_base_url",
]
