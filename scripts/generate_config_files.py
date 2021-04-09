import argparse
import os
from ada.utils.config_file_generation import ConfigVariants, Iter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate config files")

    parser.add_argument(
        "-o",
        "--output",
        help="output directory path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), os.pardir, "configs"),
    )

    args = parser.parse_args()

    ConfigVariants().add(
        "{method}", method=Iter("Source", "DANN", "CDAN-E", "WDGRL", "DAN", "JAN")
    ).add("{method}", method="CDAN", use_random=False).add(
        "{method} (rd={random_dim})",
        method="CDAN-E",
        use_random=True,
        random_dim=Iter(
            512,
            128,
        ),
    ).save(
        os.path.join(args.output, "method_variants.json")
    )
