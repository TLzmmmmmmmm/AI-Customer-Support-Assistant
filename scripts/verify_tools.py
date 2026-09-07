from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from knowledge_pipeline.retrieval import RetrievalError
from services.retrieval import build_retriever
from services.tools import build_deterministic_tools
from support_tools import ToolError, ToolErrorCode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call deterministic support tools without an LLM.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    product = subparsers.add_parser("product", help="Get exact product details.")
    product.add_argument("product_id")

    subparsers.add_parser("contact", help="Get authoritative contact data.")

    search = subparsers.add_parser(
        "search",
        help="Search products using the configured embedding provider.",
    )
    search.add_argument("query")
    return parser.parse_args(argv)


def _print_result(result) -> None:
    print(json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    ))


def _print_error(error: ToolError) -> None:
    print(json.dumps(
        {
            "code": error.code.value,
            "message": error.message,
            "tool_name": error.tool_name,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "search":
            tools = build_deterministic_tools(retriever=build_retriever())
            result = tools.search_products(args.query)
        else:
            tools = build_deterministic_tools()
            result = (
                tools.get_product_details(args.product_id)
                if args.command == "product"
                else tools.get_contact_info()
            )
        _print_result(result)
        return 0
    except ToolError as error:
        _print_error(error)
        return 1
    except RetrievalError as error:
        safe_error = ToolError(
            code=ToolErrorCode.TOOL_UNAVAILABLE,
            message="The product search dependency is unavailable.",
            tool_name="search_products",
        )
        _print_error(safe_error)
        return 1
    except Exception as error:
        safe_error = ToolError(
            code=ToolErrorCode.TOOL_EXECUTION_ERROR,
            message="The verification command could not complete safely.",
            tool_name=args.command,
        )
        _print_error(safe_error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
