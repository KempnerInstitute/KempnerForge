"""Databricks auth for MLflow: the environment the SDK and artifact clients read.

Credentials are never read here: the Databricks SDK resolves them from the environment, a
``~/.databrickscfg`` profile or OAuth. The training backend and the eval publisher both prepare
the process environment through ``_prepare_databricks_auth``, so they authenticate alike.
"""

from __future__ import annotations

import os

from kempnerforge.metrics.logger import get_logger

logger = get_logger(__name__)

_DATABRICKS_AUTH_ROUTES = (
    "Databricks auth: env DATABRICKS_HOST and DATABRICKS_TOKEN; a ~/.databrickscfg profile via "
    "a databricks://<profile> URI or DATABRICKS_CONFIG_PROFILE; SDK OAuth"
)

_profile_mismatches_warned: set[tuple[str, str]] = set()


def _auth_hint(error: Exception) -> str:
    """Where Databricks finds credentials, appended to a failure that may be a credentials one."""
    if isinstance(error, (ImportError, TimeoutError)):
        return ""
    return f" ({_DATABRICKS_AUTH_ROUTES})"


def _databricks_profile(uri: str) -> str | None:
    """The profile a ``databricks://<profile>`` tracking URI names; None for any other URI."""
    scheme, separator, rest = uri.partition("://")
    profile = rest.partition("/")[0]
    if scheme != "databricks" or not separator or not profile or ":" in profile:
        return None
    return profile


def _prepare_databricks_auth(uri: str) -> None:
    """Make SDK and artifact calls use the URI's profile, and let the token alias stand in.

    Artifact clients build their own SDK config from the environment and never see the tracking
    URI, so a ``databricks://<profile>`` URI selects that profile via DATABRICKS_CONFIG_PROFILE
    (the profile is the first path segment; ``databricks://`` and the
    ``databricks://scope:prefix`` secret-scope form select nothing), and DATABRICKS_API_TOKEN
    stands in for an unset or empty DATABRICKS_TOKEN. Non-empty operator-set values always win;
    an empty one counts as unset, as the SDK reads it. A DATABRICKS_CONFIG_PROFILE that names
    another profile than the URI is kept with a warning, since MLflow's tracking client takes
    the URI's profile while those artifact clients take the variable's. Credentials are never
    read here: the SDK resolves them from the environment, a profile file or OAuth. A URI that
    is not a Databricks one changes nothing.
    """
    if not uri.startswith("databricks"):
        return
    profile = _databricks_profile(uri)
    if profile:
        selected = os.environ.get("DATABRICKS_CONFIG_PROFILE")
        if not selected:
            os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
        elif selected != profile:
            _warn_profile_mismatch(profile, selected)
    if not os.environ.get("DATABRICKS_TOKEN") and os.environ.get("DATABRICKS_API_TOKEN"):
        os.environ["DATABRICKS_TOKEN"] = os.environ["DATABRICKS_API_TOKEN"]


def _warn_profile_mismatch(profile: str, selected: str) -> None:
    """Warn that the URI and DATABRICKS_CONFIG_PROFILE name two profiles, once per process for
    each pair, since a harness may publish many checkpoints through the same helper."""
    if (profile, selected) in _profile_mismatches_warned:
        return
    _profile_mismatches_warned.add((profile, selected))
    logger.warning(
        f"The tracking URI selects Databricks profile {profile!r} but "
        f"DATABRICKS_CONFIG_PROFILE={selected!r}: MLflow's tracking calls (and the training "
        f"backend's experiment lookup) use {profile!r}, while SDK clients built from the "
        f"environment alone, MLflow's run-artifact uploads among them, use {selected!r}. Unset "
        f"DATABRICKS_CONFIG_PROFILE or set it to {profile!r}."
    )
