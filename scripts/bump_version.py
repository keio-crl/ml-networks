#!/usr/bin/env python3
"""
バージョン管理スクリプト

使用方法:
    python scripts/bump_version.py patch   # パッチバージョンを上げる (0.1.0 -> 0.1.1)
    python scripts/bump_version.py minor   # マイナーバージョンを上げる (0.1.0 -> 0.2.0)
    python scripts/bump_version.py major   # メジャーバージョンを上げる (0.1.0 -> 1.0.0)
"""

import re
import sys
from pathlib import Path


def get_current_version() -> str:
    """pyproject.tomlから現在のバージョンを取得"""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()
    match = re.search(r'^version = "([^"]+)"', content, re.MULTILINE)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    return match.group(1)


def bump_version(version: str, bump_type: str) -> str:
    """バージョンを上げる"""
    parts = list(map(int, version.split(".")))
    
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version}")
    
    if bump_type == "major":
        parts[0] += 1
        parts[1] = 0
        parts[2] = 0
    elif bump_type == "minor":
        parts[1] += 1
        parts[2] = 0
    elif bump_type == "patch":
        parts[2] += 1
    else:
        raise ValueError(f"Invalid bump type: {bump_type}. Use 'major', 'minor', or 'patch'")
    
    return ".".join(map(str, parts))


def update_version(new_version: str) -> None:
    """pyproject.tomlのバージョンを更新"""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()
    new_content = re.sub(
        r'^version = "[^"]+"',
        f'version = "{new_version}"',
        content,
        flags=re.MULTILINE
    )
    pyproject_path.write_text(new_content)
    print(f"Updated version in pyproject.toml to {new_version}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py [major|minor|patch]")
        sys.exit(1)
    
    bump_type = sys.argv[1].lower()
    
    if bump_type not in ["major", "minor", "patch"]:
        print("Error: bump_type must be 'major', 'minor', or 'patch'")
        sys.exit(1)
    
    try:
        current_version = get_current_version()
        print(f"Current version: {current_version}")
        
        new_version = bump_version(current_version, bump_type)
        print(f"New version: {new_version}")
        
        update_version(new_version)
        print(f"\nVersion bumped successfully!")
        print(f"Next steps:")
        print(f"  1. Review the changes: git diff pyproject.toml")
        print(f"  2. Commit: git add pyproject.toml && git commit -m 'Bump version to {new_version}'")
        print(f"  3. Tag: git tag -a v{new_version} -m 'Release v{new_version}'")
        print(f"  4. Push: git push origin main && git push origin v{new_version}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

