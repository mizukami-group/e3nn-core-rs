<h1 align="center">e3nn-core-rs</h1>

<p align="center"><b>e3nn (PyTorch) とのパリティをピン留めした、framework-free な E(3)-equivariant プリミティブの Rust 実装</b></p>

<p align="center">
  <a href="LICENSE-MIT"><img alt="License: MIT OR Apache-2.0" src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue"></a>
  <img alt="Made with Rust" src="https://img.shields.io/badge/made%20with-Rust-orange">
  <img alt="Parity: e3nn 0.6.0" src="https://img.shields.io/badge/parity-e3nn%200.6.0%20(pinned)-informational">
  <img alt="unsafe: deny" src="https://img.shields.io/badge/unsafe__code-deny-brightgreen">
</p>

<p align="center">
  <a href="README.md">English</a> | 日本語
</p>

---

**e3nn-core-rs** は、[e3nn](https://github.com/e3nn/e3nn) 相当の
E(3)-equivariant プリミティブ — irreps・実球面調和関数・テンソル積・
動径基底・SO(2)/SO(3) 演算・scatter・正規化・活性化・固定重み MLP — を
Rust で実装したライブラリです。力(forces)を出力する MLIP 推論のための
低レベル演算カーネル層で、PyTorch などのテンソルフレームワークに
依存しません(framework-free)。**学習機能と汎用 autograd エンジンは
提供しません**: 勾配は各演算の手書き VJP/JVP カーネルとして提供され、
数値的正しさは e3nn (PyTorch) 0.6.0 で生成したフィクスチャに
ピン留めされています。

本プロジェクトは独立した実装であり、e3nn 公式プロジェクトとは無関係で、
公式の承認・推奨を受けたものではありません。`e3nn-core` という名前は
数値互換性の対象を示し、e3nn 公式の Rust 版を意味しません。

`llmlip` は本ライブラリのプリミティブを利用する別の MLIP 推論エンジンです。
モデルの読み込みやモデル全体の実行は本リポジトリの対象外です。

## ✨ 特長

- 🦀 **Framework-free** — PyTorch もテンソルフレームワークも不要。
  `&[f64]` スライス上の素の Rust。
- 🎯 **パリティのピン留め** — 出力は e3nn (PyTorch) 0.6.0
  (commit `2aa7f584`)で生成したフィクスチャと照合してテスト
  (シード・dtype・規約メタデータを記録)。
- ✍️ **手書き勾配** — 対応する演算に明示的な VJP/JVP カーネルを提供します。
- 🔒 **`unsafe_code = "deny"`** — 純 Rust 実装(例外は opt-in の
  BLAS フィーチャ配下に隔離された FFI モジュール)。

## 📦 モジュール構成

`crates/e3nn-core` の公開モジュール(`src/lib.rs` 準拠):

| モジュール | 内容 |
|---|---|
| `irreps` | O(3) 既約表現(`Irrep` / `IrrepMul` / `Irreps` / `Parity`)の解析・次元計算・レイアウト(`LayoutSegment`) |
| `sh` | 実球面調和関数(`SphericalHarmonicsPlan`、component 正規化、`l_max <= 4`)と VJP/JVP |
| `tp` | テンソル積(`TensorProductPlan` / `TensorProductInstruction` / `CgEntry`) |
| `radial` | 動径基底・カットオフ関数 |
| `so2` / `so3` | SO(2)/SO(3) 演算(eSEN 系 SO(2) 畳み込み・Wigner 回転など) |
| `scatter` | エッジ→ノード集約(scatter)演算 |
| `normalization` | 正規化カーネル(RMS norm など) |
| `activation` | ゲート活性化・S2 活性化カーネル |
| `mlp` | 固定重み MLP(`FixedMlp` / `DenseLayer` / `Activation` / `MlpCache`) |
| `geometry` | `Vec3`(`[f64; 3]`)とエッジベクトル等の幾何ユーティリティ |
| `error` | `CoreError` / `Result` |

## 🧪 設計: ピン留めフィクスチャによるパリティ検証

このプロジェクトは規約(convention)に敏感です。符号・成分順・正規化が
1 つでもずれると、モデル重みの移植には使えません。そのため:

- 参照実装 **e3nn (PyTorch) 0.6.0**(commit `2aa7f584`)で生成した
  フィクスチャを `crates/e3nn-core/fixtures/e3nn_torch/v0_6_0_2aa7f584/` にピン留めし、
  Rust 実装の出力を FP64 でテスト照合します。
- 規約は記憶に頼らずフィクスチャで検証します。規約を変更する場合は
  フィクスチャを再生成し、
  [docs/MATH_CONVENTIONS.md](docs/MATH_CONVENTIONS.md) を更新します。
- Rust ランタイムは Python に依存しません(フィクスチャ生成のみ Python)。

詳細は [docs/MATH_CONVENTIONS.md](docs/MATH_CONVENTIONS.md)・
[docs/TESTING.md](docs/TESTING.md)・
[docs/DERIVATIVES.md](docs/DERIVATIVES.md) を参照して
ください。

## 🚀 使用例

```rust
use e3nn_core::{Irreps, SphericalHarmonicsPlan, Vec3};

fn main() -> e3nn_core::Result<()> {
    // Irreps の解析と次元・レイアウト
    let irreps: Irreps = "16x0e + 16x1o + 8x2e".parse()?;
    assert_eq!(irreps.dim(), 16 + 48 + 40);

    // 実球面調和関数(component 正規化、l_max <= 4)
    let plan = SphericalHarmonicsPlan::component(2, true)?;
    let r: Vec3 = [0.1, -0.4, 0.9];
    let mut y = vec![0.0_f64; plan.checked_output_dim()?];
    plan.forward(r, &mut y)?;
    assert_eq!(y.len(), 9); // (l_max + 1)^2

    Ok(())
}
```

勾配が必要な場合は `SphericalHarmonicsPlan::vjp_edge_vector` /
`jvp_edge_vector` など、各演算に対応する VJP/JVP カーネルを使います。
テンソル積は `TensorProductPlan::new` でプランを構築し
`forward` / `forward_accumulate` で評価します。

## 🛠 ビルドとテスト

```bash
cargo build
cargo test            # フィクスチャ照合テストを含む(fixtures/ を同梱)
cargo bench           # criterion ベンチ(radial / sh / tensor_product)
```

Cargo フィーチャ(デフォルトはすべて無効):

| フィーチャ | 内容 |
|---|---|
| `parallel` | rayon による並列化 |
| `accelerate-blas` | macOS の Accelerate BLAS を一部 SO(3) 経路で使用(macOS 以外では純 Rust 経路にフォールバック) |
| `openblas` | Linux x86_64 の OpenBLAS（任意で `OPENBLAS_LIB_DIR` を指定） |
| `intel-mkl` | Linux x86_64 の Intel oneMKL を逐次実行・静的リンク（`MKLROOT`） |
| `armpl` | Linux aarch64 向け ArmPL マーカー。`openblas` 呼び出しを有効化し、利用側で ArmPL をリンク。SO(3) はスカラー経路 |
| `serde` | 一部型の serde 対応 |

BLAS プロバイダーは一度に 1 つ選択してください。`intel-mkl` を有効にした
バイナリを配布する場合、利用した Intel oneMKL に同梱されるライセンスと
通知の同梱条件に従ってください（[Intel oneMKL license FAQ](https://www.intel.com/content/www/us/en/developer/articles/tool/onemkl-license-faq.html)、
Intel Simplified Software License）。Rust ソースのライセンスは、リンクする
ネイティブライブラリのライセンスを置き換えるものではありません。

## 🧷 API 安定性について

このクレートは pre-1.0 であり、API はまだ安定していません。特に、
`fairchem_esen_*` / `mace_*` といったモデル由来の名前を持つカーネル
(`activation` / `normalization` / `radial` / `scatter` / `so2` / `so3`
モジュール内)は、特定モデルアーキテクチャの数値挙動を再現するための
**experimental / unstable API** です。これらについて SemVer 上の安定性は
約束しません。リネーム・統合・削除がマイナーバージョンで発生し得ます。

## ⚖️ ライセンス

**MIT** または **Apache-2.0** のいずれか任意の方を選択できます
([LICENSE-MIT](LICENSE-MIT) / [LICENSE-APACHE](LICENSE-APACHE))。
本リポジトリにはモデルの学習済み重みは含まれません。同梱するのは
テスト用の数値参照フィクスチャ(JSON)のみです。

## 👥 Contributors

This project was developed by:

- Wataru Mizukami ([@wmizukami](https://github.com/wmizukami))
- Do Tuan Minh ([@URPhysChem](https://github.com/URPhysChem))
- Kenji Ishihara ([@kenji-ishihara-os](https://github.com/kenji-ishihara-os))

## 🙏 謝辞

数値参照実装として [e3nn](https://github.com/e3nn/e3nn)、
[MACE](https://github.com/ACEsuit/mace)、
[fairchem（eSEN/UMA）](https://github.com/FAIR-Chem/fairchem) に謝意を表します。
動径カーネルは ZBL 遮蔽モデル（Ziegler・Biersack・Littmark、1985）、
[Cordero ら（2008）](https://doi.org/10.1039/B801115J) に基づく
[ASE](https://ase.gitlab.io/ase/ase/data.html) の共有結合半径表を利用する MACE の規約、
および MACE の Agnesi 変換規約を参照しています。バージョン、学術的出典、
上流ライセンス全文は [third-party notices](THIRD_PARTY_NOTICES.md) を参照してください。
これらの謝辞は学習済みモデル重みの利用許諾を与えるものではありません。

This work is supported by JST COI-NEXT「量子ソフトウェア研究拠点」
(Quantum Software Research Hub), Grant Number **JPMJPF2014**.

## Local validation

Rust 1.89 or newer is required. No GitHub Actions workflows are provided.
Run `bash scripts/check.sh` locally; see [testing](docs/TESTING.md).
Linux-specific `openblas`, `intel-mkl`, and `armpl` features require their
native libraries and are not validated by the macOS checks. Select one BLAS
provider at a time.

External contributions are not currently accepted.
