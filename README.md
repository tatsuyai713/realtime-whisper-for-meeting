# Meeting Live Translator

Mac / Windows の会議音声を取り込み、左側に原文、右側に少し遅れて日本語訳を
表示する Python デスクトップアプリです。無音を検出して発言を区切るため、
会話のまとまりごとに自動改行されます。

## 処理方式と使用モデル

- OpenAIモード
  - 文字起こし: `gpt-4o-transcribe`
  - 日本語翻訳: `gpt-5.6-sol`
- Localモード
  - Apple Silicon Mac（高速・既定）: `WhisperKit` + Core ML / Neural Engine
  - Apple Silicon Mac（代替）: `mlx-whisper` + Metal GPU
  - Windows: `faster-whisper` + CUDA/CPU
  - モデル: `tiny` / `base` / `small` / 圧縮`small` / `turbo` / `medium` / `large-v3`
  - 日本語翻訳: 翻訳専用の `translategemma:12b`（画面で変更可能）

日本語翻訳は画面からON/OFFできます。OpenAIモードではAPI使用料金がかかります。
Localモードでは音声もテキストもPC外へ送信しません。
ただし「別PC（翻訳サーバー）」を選択した場合、文字起こし結果は指定したLAN内の
翻訳サーバーへ送信されます。音声データは送信されません。
Apple Siliconでは画面の「高速化」でNPU（Neural Engine）またはGPU（Metal）を
選択できます。M4 Macでは専用の高速化プリセットも表示されます。ローカル処理では
選択したプリセットに応じて音声を最大3.5～5秒で区切り、境界の0.4～0.8秒を重ねて
単語の途中切れを防ぎます。発話開始前の0.3秒も保持し、小さい語頭音を取りこぼしません。
WhisperKitの単語タイムスタンプで重複音声部分だけを除外し、表記揺れがある場合も
補助的な類似判定で重複文字を自動除去します。日本語・中国語・韓国語では単語途中を
切らないようタイムスタンプ切除を避け、強制分割された区間を同じ発話として境界修復
してから表示します。英語でも前区間に同じ語が確認できた場合だけ重複部分を除外し、
新しく認識された語は保持します。表示時は `.`、`?`、`!`、`。`、`？`、`！` ごとに
自動改行します。
NPUを選ぶと文字起こしとOllama翻訳がGPUを奪い合わず、並行処理しやすくなります。
M4 Macの既定値は「高速（圧縮small）」+ Neural Engineです。モデルは初回使用時だけ
自動取得されます。処理後はステータス欄に `RTF` と実時間に対する処理倍率を表示します。

### M4最適化プリセット

- 高速（圧縮small）: WhisperKitの`small_216MB`、5秒チャンク、0.4秒オーバーラップ
- バランス（Turbo 4bit）: `large-v3-v20240930_turbo_632MB`、5秒チャンク
- 高精度（large-v3）: 非圧縮`large-v3-v20240930`、4.5秒チャンク
- カスタム: モデルと処理エンジンに加え、Neural Engine、ANE + GPU、GPUを指定可能

初回使用時はCore MLモデルの取得と端末向けコンパイルが行われます。高速化の比較には
ステータス欄のRTFを使用します。RTFが小さいほど高速で、例えばRTF 0.20は音声の長さに
対して約5倍速です。

## セットアップ

Python 3.10 以上を用意し、このフォルダで次を実行します。

### macOS / Linux shell

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python run.py
```

Localモードも使う場合は、次を追加で実行します。

```bash
python -m pip install -e ".[local]"
brew install whisperkit-cli
```

`whisperkit-cli` はApple Silicon MacのNeural Engine用です。初回に選んだ音声モデルを
取得し、Core MLが端末向けにコンパイルするため数分かかることがあります。2回目以降は
キャッシュを利用します。Windowsではこの手順は不要です。

ローカル翻訳を使う場合は [Ollama](https://ollama.com/) をインストールし、先に
モデルを取得してください。

```bash
ollama pull translategemma:12b
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
python run.py
```

Localモードを使う場合は、PowerShellでも次を追加実行します。

```powershell
python -m pip install -e ".[local]"
ollama pull translategemma:12b
```

APIキーは画面に入力するか、起動前に `OPENAI_API_KEY` 環境変数へ設定できます。
画面に入力したキーは保存されません。

## 別PCで日本語翻訳を実行する

文字起こしはMac側で実行し、日本語翻訳だけをLAN内の別マシン上のOllamaへ
任せることができます。推奨サーバーはUbuntuとAMD Radeon AI PRO R9700です。
翻訳サーバーは既定でGPU必須となり、ROCmまたはOllamaがCPUへフォールバックしている
場合は起動しません。

### Ubuntu + Radeon AI PRO R9700（推奨）

R9700はRDNA4の`gfx1201`としてROCmとOllamaのLinux版で正式対応されています。
ROCmの対応OSを厳密に合わせる必要があるため、Ubuntu 24.04.4または22.04.5を
使用してください。以下はUbuntu 24.04系、ROCm 7.2.4を例にしています。バージョンを
変更する場合は[AMDの対応表](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.3/reference/system-requirements.html)と
[ROCm Quick Start](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.4/install/quick-start.html)を確認してください。

#### 1. ROCmドライバーをインストール

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb
sudo apt install ./amdgpu-install_7.2.4.70204-1_all.deb
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms rocm
sudo usermod -aG render,video "$USER"
sudo reboot
```

再起動後、R9700と`gfx1201`が表示されることを確認します。

```bash
rocminfo | grep -E "Marketing Name|gfx1201"
ls -l /dev/kfd /dev/dri/renderD*
```

#### 2. Ollamaをインストール

[OllamaのLinux手順](https://github.com/ollama/ollama/blob/main/docs/linux.mdx)に従い、
最新版とAMD ROCmランタイムをインストールします。

```bash
if command -v ollama >/dev/null 2>&1 && \
   find /usr/lib/ollama /usr/local/lib/ollama -maxdepth 1 \
     -type d -name 'rocm*' -print -quit 2>/dev/null \
     | grep -q .; then
  echo "Ollama AMD/ROCm版は導入済みです。ダウンロードをスキップします。"
else
  sudo apt-get update
  sudo apt-get install -y zstd
  curl -fsSL https://ollama.com/install.sh | sh
fi
sudo usermod -aG render,video ollama
sudo systemctl restart ollama

if ollama show translategemma:12b >/dev/null 2>&1; then
  echo "translategemma:12bは取得済みです。ダウンロードをスキップします。"
else
  ollama pull translategemma:12b
fi
```

公式インストーラーが`AMD GPU ready.`と表示した場合、AMD ROCmランタイムの導入も
完了しているため、ROCmアーカイブを手動で再展開する必要はありません。
手動インストール時に`tar: Archive is compressed. Use --zstd option`と表示された場合は、
`zstd`を介して展開します。

```bash
if find /usr/lib/ollama /usr/local/lib/ollama -maxdepth 1 \
     -type d -name 'rocm*' -print -quit 2>/dev/null \
     | grep -q .; then
  echo "ROCmランタイムは導入済みです。ダウンロードをスキップします。"
else
  curl -fsSL https://ollama.com/download/ollama-linux-amd64-rocm.tar.zst \
    | zstd -d | sudo tar -xf - -C /usr
fi
sudo systemctl restart ollama
```

簡単な推論を一度実行し、`PROCESSOR`が`100% GPU`またはGPUを含む表示になることを
確認します。CPUだけが表示された状態では本プロジェクトのサーバーは起動しません。

```bash
ollama run translategemma:12b "Translate Hello into Japanese."
ollama ps
```

OllamaはR9700を`gfx1201`として正式サポートしており、LinuxではROCm v7ドライバーが
必要です。[Ollama AMD GPU対応表](https://github.com/ollama/ollama/blob/main/docs/gpu.mdx)
古いROCm 6系ドライバーではGPU検出がタイムアウトしてCPUへフォールバックするため、
`HSA_OVERRIDE_GFX_VERSION`で回避せずROCm v7へ更新してください。

#### 3. 翻訳サーバーを起動

```bash
git clone git@github.com:tatsuyai713/realtime-whisper-for-mac.git
cd realtime-whisper-for-mac
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e . --no-deps
python run_translation_server.py --check-gpu
python run_translation_server.py \
  --host 0.0.0.0 \
  --port 8765 \
  --model translategemma:12b
```

`--check-gpu`は`rocminfo`、`gfx1201`、`/dev/kfd`のアクセス権を検査します。通常起動時は
さらに指定モデルをOllamaへロードし、`/api/ps`の`size_vram`が0より大きいことを確認して
から待受を開始します。GPUなしでの動作を意図的に許可する場合だけ`--allow-cpu`を
指定できます。

別の待受IP、ポート、Ollamaを指定する例です。

```bash
python run_translation_server.py \
  --host 192.168.1.50 --port 9000 \
  --ollama-host 127.0.0.1 --ollama-port 11434 \
  --model translategemma:12b
```

UFWを使用している場合は、Macが接続しているLANからの通信だけを許可します。

```bash
sudo ufw allow from 192.168.1.0/24 to any port 8765 proto tcp
```

認証や暗号化は行わないため、インターネットへ直接公開しないでください。

### Windowsでサーバーを動かす場合

R9700の推奨構成はUbuntu + ROCmです。WindowsやGPU診断非対応環境では、明示的に
`--allow-cpu`を指定した場合だけ起動できます。

```powershell
python run_translation_server.py --host 0.0.0.0 --port 8765 --allow-cpu
```

### Mac側（GUI）

1. 文字起こしに使う処理方式（OpenAIまたはLocal）を選択する
2. 「日本語翻訳」をONにする
3. 「翻訳実行」で「別PC（翻訳サーバー）」を選択する
4. 翻訳用マシンのIPアドレスまたはホスト名とポートを入力する
5. 「開始」を押す

開始時にサーバーへの接続と翻訳モデルの有無を確認します。既定ポートは `8765` です。
環境変数 `TRANSLATION_TARGET=server`、`TRANSLATION_SERVER_HOST`、
`TRANSLATION_SERVER_PORT` でもGUIの初期値を設定できます。

## システム音声の選び方

### Windows

「取得する音声」一覧から名前に「システム音声」と付いたスピーカーを選びます。
WASAPIループバックにより、そのスピーカーから再生されるYouTube、会議アプリ、
ブラウザなどの音声を取得します。
表示されない場合は Windows のサウンド設定で「ステレオ ミキサー」を有効にして、
アプリの「再読込」を押してください。

### macOS

一覧の先頭にある「Macで再生中の音声」を選びます。Apple ScreenCaptureKitにより、
YouTube、会議アプリ、ブラウザなどの再生音を直接取得するため、BlackHoleは不要です。
初回はmacOSの「画面収録とシステムオーディオ」権限を許可し、アプリを再起動して
ください。拒否済みの場合は、画面右下の「Mac音声権限」を押し、一覧に表示される
Python、Terminal、または `macos-system-audio-capture` を有効にします。

音声ヘルパーの初回ビルドにはApple Command Line Toolsが必要です。

```bash
xcode-select --install
```

BlackHole等の仮想入力も引き続き一覧から選択できます。

## 操作

1. 「システム音声」と表示されたデバイスを選ぶ
2. 発話言語（または自動検出）を選ぶ
3. OpenAIまたはLocalを選ぶ
4. M4 MacのLocalモードでは、目的に合う「M4最適化」を選ぶ
5. 必要に応じて日本語翻訳をON/OFFする
6. OpenAIの場合はAPIキーを入力する
7. 「開始」を押す
8. 終了後、「保存」で原文と日本語訳をテキスト保存する

音声は約8秒を上限に、無音の位置を優先して区切られます。OpenAIモードでは音声と
文字列がOpenAI APIへ送信され、LocalモードではPC内だけで処理されます。機密会議
では、所属組織の情報管理規程と参加者の同意を確認してください。

## 開発用チェック

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```
