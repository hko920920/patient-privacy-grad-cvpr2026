"""Registry of accountant routes used to keep claim boundaries explicit."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AccountantRouteSpec:
    route_id: str
    sampling_unit: str
    accounting_unit: str
    status: str
    certificate_use: str
    required_assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ACCOUNTANT_ROUTES = (
    AccountantRouteSpec(
        route_id="owa_owner_rdp",
        sampling_unit="owner",
        accounting_unit="owner",
        status="legacy_not_release_validated",
        certificate_use="legacy research diagnostics only; not a privacy release",
        required_assumptions=(
            "independent Bernoulli owner sampling",
            "single-attribution selected windows",
            "one clipped vector per sampled owner per step",
        ),
    ),
    AccountantRouteSpec(
        route_id="window_rdp_group_fallback",
        sampling_unit="window",
        accounting_unit="owner",
        status="implemented_conservative",
        certificate_use="valid but loose owner fallback",
        required_assumptions=(
            "valid window-level DP-SGD accountant",
            "selected owner multiplicity bound",
            "standard group privacy conversion",
        ),
    ),
    AccountantRouteSpec(
        route_id="bounded_multiplicity_els_mixture",
        sampling_unit="window",
        accounting_unit="owner",
        status="implemented_limited",
        certificate_use="limited UCI baseline and diagnostic",
        required_assumptions=(
            "independent window Poisson sampling",
            "selected owner multiplicity bound",
            "worst-aligned owner contribution mixture",
            "configured RDP orders and quadrature",
        ),
    ),
    AccountantRouteSpec(
        route_id="mog_pld_els_owner",
        sampling_unit="window",
        accounting_unit="owner",
        status="legacy_not_execution_bound",
        certificate_use="legacy diagnostic; sampler/executor binding not validated",
        required_assumptions=(
            "independent window Poisson sampling",
            "selected owner multiplicity bound",
            "dp_accounting Mixture-of-Gaussians PLD discretization",
            "sampler and contribution model match the selected mapping",
        ),
    ),
    AccountantRouteSpec(
        route_id="fixed_size_mog_pld_els_owner",
        sampling_unit="window",
        accounting_unit="owner",
        status="legacy_not_execution_bound",
        certificate_use="legacy diagnostic; sampler/executor binding not validated",
        required_assumptions=(
            "fixed-size window sampling without replacement",
            "window population size and batch size match the accountant",
            "selected owner multiplicity bound",
            "Hypergeometric owner-change count model",
            "dp_accounting Mixture-of-Gaussians PLD discretization",
        ),
    ),
    AccountantRouteSpec(
        route_id="external_tight_els_mog",
        sampling_unit="window",
        accounting_unit="owner",
        status="external_hook",
        certificate_use="not used by current certificates",
        required_assumptions=(
            "audited tight ELS/MoG implementation",
            "sampler matches accountant",
            "multiplicity and contribution model match the mapping",
        ),
    ),
    AccountantRouteSpec(
        route_id="external_fixed_size_els_owner",
        sampling_unit="window",
        accounting_unit="owner",
        status="external_hook",
        certificate_use="not used by current certificates",
        required_assumptions=(
            "alternative audited fixed-size or without-replacement ELS owner accountant",
            "window batch size and source population match the accountant",
            "selected owner multiplicity or contribution model is bounded",
            "adjacency convention and composition schedule are declared",
            "runtime sampler trace proves fixed-size window sampling law",
        ),
    ),
    AccountantRouteSpec(
        route_id="external_shuffled_epoch_els_owner",
        sampling_unit="window",
        accounting_unit="owner",
        status="external_hook",
        certificate_use="not used by current certificates",
        required_assumptions=(
            "audited shuffled or epoch-level ELS owner accountant",
            "loader implements the shuffled/epoch law assumed by the accountant",
            "selected owner multiplicity or contribution model is bounded",
            "epoch composition and adjacency convention are declared",
            "runtime trace records epoch order or a verifiable digest",
        ),
    ),
    AccountantRouteSpec(
        route_id="external_owner_fixed_size_rdp",
        sampling_unit="owner",
        accounting_unit="owner",
        status="implemented_probe",
        certificate_use="SRSWOR replace-one owner probe; not used by main certificates",
        required_assumptions=(
            "fixed-size owner batches sampled without replacement (SRSWOR)",
            "dp_accounting RDP under replace-one neighboring relation",
            "Gaussian noise calibrated to 2C replace-one step sensitivity",
            "single-attribution selected windows",
            "one clipped vector per sampled owner per step",
        ),
    ),
    AccountantRouteSpec(
        route_id="external_owner_shuffle_epoch",
        sampling_unit="owner",
        accounting_unit="owner",
        status="implemented_probe",
        certificate_use="conservative shuffled-epoch owner probe; not used by main certificates",
        required_assumptions=(
            "owner epochs are shuffled and partitioned into fixed-size batches",
            "per-batch dp_accounting SRSWOR RDP under replace-one neighboring relation",
            "composition is conservative and does not claim tight epoch-shuffle amplification",
            "epoch construction matches the implemented owner loader",
            "single-attribution selected windows",
            "one clipped vector per owner visit before accounting",
        ),
    ),
    AccountantRouteSpec(
        route_id="external_structured_event",
        sampling_unit="window",
        accounting_unit="event",
        status="external_hook",
        certificate_use="not used by current certificates",
        required_assumptions=(
            "structured event-level accountant",
            "event multiplicity model matches selected windows",
            "sampler matches accountant",
        ),
    ),
)


def accountant_routes() -> list[dict[str, object]]:
    return [route.to_dict() for route in ACCOUNTANT_ROUTES]


def get_accountant_route(route_id: str) -> AccountantRouteSpec:
    for route in ACCOUNTANT_ROUTES:
        if route.route_id == route_id:
            return route
    raise KeyError(route_id)
