from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from project_mujoco_visualizer.motion_loader import load_motion_csv, load_motion_npz


class RepositoryDataTests(unittest.TestCase):
    def test_real_csv_joint_names_match_robot_xml(self) -> None:
        root = Path(__file__).resolve().parents[1]
        motion = load_motion_csv(root / "data" / "666_Dance.csv")
        xml_path = root / "robot_asset" / "roban_s22_handball" / "xml" / "biped_s17_verified_handball_fixedhead.xml"
        xml_root = ET.parse(xml_path).getroot()
        xml_joint_names = {
            element.attrib["name"]
            for element in xml_root.iter("joint")
            if element.attrib.get("type") != "free" and "name" in element.attrib
        }
        self.assertEqual(set(motion.joint_names), xml_joint_names)
        self.assertEqual(len(motion.joint_names), 21)

    def test_real_lafan_npz_schema_and_joint_names(self) -> None:
        root = Path(__file__).resolve().parents[1]
        motion = load_motion_npz(root / "data" / "LAFAN" / "lafan_s22_npz" / "walk1_subject1.npz")
        xml_path = root / "robot_asset" / "roban_s22_handball" / "xml" / "biped_s17_verified_handball_fixedhead.xml"
        xml_root = ET.parse(xml_path).getroot()
        xml_joint_names = {
            element.attrib["name"]
            for element in xml_root.iter("joint")
            if element.attrib.get("type") != "free" and "name" in element.attrib
        }
        self.assertEqual(motion.frame_count, 13066)
        self.assertAlmostEqual(motion.sample_rate_hz, 50.0)
        self.assertEqual(set(motion.joint_names), xml_joint_names)
        self.assertEqual(motion.root_quaternion.shape, (13066, 4))


if __name__ == "__main__":
    unittest.main()
