# 開発

開発に関する情報を説明します。

## 開発環境のセットアップ

```bash
# リポジトリをクローン
git clone https://github.com/keio-crl/ml-networks.git
cd ml-networks

# 開発用依存関係をインストール
pip install -e ".[dev]"
```

## コード品質チェック

```bash
# リンターの実行
ruff check .

# 型チェック
mypy src/

# フォーマット
ruff format .
```

## ドキュメントのビルド

```bash
# ドキュメント用依存関係をインストール
pip install -e ".[docs]"

# ドキュメントをビルド
mkdocs build

# ローカルサーバーでドキュメントを確認
mkdocs serve
```

## バージョン管理

このプロジェクトでは、セマンティックバージョニング（Semantic Versioning）を使用しています。

### バージョンの形式

バージョンは `MAJOR.MINOR.PATCH` の形式で管理されています：
- **MAJOR**: 互換性のない変更がある場合
- **MINOR**: 後方互換性を保った機能追加の場合
- **PATCH**: 後方互換性を保ったバグ修正の場合

### バージョンの更新方法

#### 方法1: スクリプトを使用（推奨）

```bash
# パッチバージョンを上げる (0.1.0 -> 0.1.1)
python scripts/bump_version.py patch

# マイナーバージョンを上げる (0.1.0 -> 0.2.0)
python scripts/bump_version.py minor

# メジャーバージョンを上げる (0.1.0 -> 1.0.0)
python scripts/bump_version.py major
```

スクリプト実行後、以下の手順でリリースします：

```bash
# 変更を確認
git diff pyproject.toml

# コミット
git add pyproject.toml
git commit -m "Bump version to X.Y.Z"

# タグを作成
git tag -a vX.Y.Z -m "Release vX.Y.Z"

# プッシュ
git push origin main
git push origin vX.Y.Z
```

#### 方法2: GitHub Actionsを使用

1. GitHubリポジトリの「Actions」タブに移動
2. 「Version Bump」ワークフローを選択
3. 「Run workflow」をクリック
4. バージョンタイプ（patch/minor/major）を選択
5. タグを作成するかどうかを選択
6. ワークフローを実行

## リリースプロセス

1. **バージョンを更新**: 上記の方法でバージョンを更新
2. **タグを作成**: `vX.Y.Z` の形式でタグを作成
3. **自動リリース**: タグをプッシュすると、GitHub Actionsが自動的に：
   - パッケージをビルド
   - リリースノートを生成
   - GitHub Releaseを作成

## CI/CD

このプロジェクトでは、以下のGitHub Actionsワークフローが設定されています：

- **CI**: プッシュ/プルリクエスト時に自動実行
  - リントチェック（ruff）
  - 型チェック（mypy）
  - テスト実行（pytest）
  - パッケージビルド確認

- **Release**: タグがプッシュされたときに自動実行
  - パッケージのビルド
  - GitHub Releaseの作成

## コントリビューション

コントリビューションを歓迎します。プルリクエストを送信する前に、コード品質チェックを実行してください。
