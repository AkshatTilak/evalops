"""EvalOps dataset manager package."""

from projects.evalops.src.datasets.manager import (
    create_suite,
    list_suites,
    get_suite,
    update_suite,
    delete_suite,
    clone_suite,
    add_test_case,
    update_test_case,
    delete_test_case,
    list_test_cases,
    import_cases_from_csv,
    import_cases_from_json,
    export_suite_to_json,
)

__all__ = [
    "create_suite",
    "list_suites",
    "get_suite",
    "update_suite",
    "delete_suite",
    "clone_suite",
    "add_test_case",
    "update_test_case",
    "delete_test_case",
    "list_test_cases",
    "import_cases_from_csv",
    "import_cases_from_json",
    "export_suite_to_json",
]
