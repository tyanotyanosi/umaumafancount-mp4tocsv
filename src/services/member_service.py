import json
from pathlib import Path
from typing import Dict
from src.domain.models import MemberEntry

class MemberService:
    """メンバーリストと置換JSONの読み込み・書き出し"""

    def load(self, list_path: Path, replace_path: Path) -> Dict[str, MemberEntry]:
        members = {}
        # memberList.txt を読む
        if list_path.exists():
            with open(list_path, 'r', encoding='utf-8') as f:
                for line in f:
                    name = line.strip()
                    if name:
                        members[name] = MemberEntry(name=name)

        # memberReplace.json の置換パターンをマージ
        if replace_path.exists():
            with open(replace_path, 'r', encoding='utf-8') as f:
                try:
                    replace_data = json.load(f)
                except json.JSONDecodeError:
                    replace_data = {}
                
            for name, patterns in replace_data.items():
                if name in members and isinstance(patterns, list):
                    members[name].replace_patterns = patterns
        
        return members

    def save(self, members: Dict[str, MemberEntry], list_path: Path, replace_path: Path):
        """memberList.txt と memberReplace.json を両方書き出す"""
        list_path.parent.mkdir(parents=True, exist_ok=True)

        # memberList.txt を書き出し
        with open(list_path, 'w', encoding='utf-8') as f:
            for entry in members.values():
                f.write(f"{entry.name}\n")

        # memberReplace.json を書き出し（空パターンは除外）
        replace_data = {}
        for entry in members.values():
            if entry.replace_patterns:
                replace_data[entry.name] = entry.replace_patterns

        with open(replace_path, 'w', encoding='utf-8') as f:
            json.dump(replace_data, f, ensure_ascii=False, indent=2)
