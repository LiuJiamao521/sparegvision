from __future__ import annotations

import argparse
import importlib
import sys


COMMANDS = {
    "simulate": "sparegvision.run_simulation",
    "complexity-simulation": "sparegvision.run_complexity_simulation",
    "gw12-screen": "sparegvision.run_gw12_genomewide_complexity",
    "gw12-rank": "sparegvision.plot_gw12_genomewide_srci",
    "zebrafish-screen": "sparegvision.run_zebrafish_genomewide_complexity",
    "zebrafish-confirm": "sparegvision.run_zebrafish_confirm_complexity",
    "zebrafish-rank": "sparegvision.plot_zebrafish_genomewide_srci",
    "zebrafish-gene-figures": "sparegvision.plot_zebrafish_complexity_genes",
    "zebrafish-domains": "sparegvision.plot_zebrafish_spatial_domains",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="sparegvision",
        description="Spatial multi-enhancer regulatory complexity and attribution",
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    module = importlib.import_module(COMMANDS[parsed.command])
    previous = sys.argv
    try:
        sys.argv = [COMMANDS[parsed.command], *parsed.args]
        module.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
