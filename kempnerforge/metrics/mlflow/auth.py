"""Databricks auth for MLflow: the environment the SDK and artifact clients read.

Credentials are never read here: the Databricks SDK resolves them from the environment, a
``~/.databrickscfg`` profile or OAuth. The training backend and the eval publisher both prepare
the process environment through ``_prepare_databricks_auth``, so they authenticate alike.
"""

from __future__ import annotations

import os

_DATABRICKS_AUTH_ROUTES = (
    "Databricks auth: env DATABRICKS_HOST and DATABRICKS_TOKEN; a ~/.databrickscfg profile via "
    "a databricks://<profile> URI or DATABRICKS_CONFIG_PROFILE; SDK OAuth"
)


def _auth_hint(error: Exception) -> str:
    """Where Databricks finds credentials, appended to a failure that may be a credentials one."""
    if isinstance(error, (ImportError, TimeoutError)):  # a missing package or a slow server
        return ""
    return f" ({_DATABRICKS_AUTH_ROUTES})"


def _databricks_profile(uri: str) -> str | None:
    """The profile a ``databricks://<profile>`` tracking URI names; None for any other URI."""
    scheme, separator, rest = uri.partition("://")
    profile = rest.partition("/")[0]
    if scheme != "databricks" or not separator or not profile or ":" in profile:
        return None  # "scope:prefix" names a secret scope, not a profile
    return profile


def _prepare_databricks_auth(uri: str) -> None:
    """Make SDK and artifact calls use the URI's profile, and let the token alias stand in.

    Artifact clients build their own SDK config from the environment and never see the tracking
    URI, so a ``databricks://<profile>`` URI selects that profile via DATABRICKS_CONFIG_PROFILE
    (the profile is the first path segment; ``databricks://`` and the
    ``databricks://scope:prefix`` secret-scope form select nothing), and DATABRICKS_API_TOKEN
    stands in for an unset or empty DATABRICKS_TOKEN. Non-empty operator-set values always win.
    Credentials are never read here: the SDK resolves them from the environment, a profile file
    or OAuth. A URI that is not a Databricks one changes nothing.
    """
    if not uri.startswith("databricks"):
        return
    profile = _databricks_profile(uri)
    if profile:
        os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    # The Databricks SDK reads DATABRICKS_TOKEN; mirror the DATABRICKS_API_TOKEN
    # name many keep in ~/.bashrc so either works.
    if not os.environ.get("DATABRICKS_TOKEN") and os.environ.get("DATABRICKS_API_TOKEN"):
        os.environ["DATABRICKS_TOKEN"] = os.environ["DATABRICKS_API_TOKEN"]
