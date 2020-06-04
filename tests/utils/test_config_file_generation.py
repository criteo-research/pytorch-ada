import unittest
from ada.utils.config_file_generation import ConfigVariants, Iter


class TestConfigVariants(unittest.TestCase):
    def test_config_variants(self):
        cv = (
            ConfigVariants()
            .add("CDAN", method="CDAN", use_random=False)
            .add(
                "{method} (rd={random_dim})",
                method="CDAN",
                use_random=True,
                random_dim=4,
            )
        )
        self.assertEqual(
            {
                "CDAN": {"method": "CDAN", "use_random": False},
                "CDAN (rd=4)": {"method": "CDAN", "use_random": True, "random_dim": 4},
            },
            cv.to_dict(),
        )
        cv.add(
            "{method} (rd={random_dim})",
            method=Iter("DANN", "CDAN"),
            use_random=True,
            random_dim=Iter(range(3, 5)),
        )
        self.assertEqual(
            {
                "CDAN": {"method": "CDAN", "use_random": False},
                "CDAN (rd=4)#0": {
                    "method": "CDAN",
                    "use_random": True,
                    "random_dim": 4,
                },
                "DANN (rd=3)": {"method": "DANN", "use_random": True, "random_dim": 3},
                "DANN (rd=4)": {"method": "DANN", "use_random": True, "random_dim": 4},
                "CDAN (rd=3)": {"method": "CDAN", "use_random": True, "random_dim": 3},
                "CDAN (rd=4)#1": {
                    "method": "CDAN",
                    "use_random": True,
                    "random_dim": 4,
                },
            },
            cv.to_dict(),
        )
