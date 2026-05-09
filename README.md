# MIGE

Manual Infant Gaze Encoder

Michael Makoto Martinsen

MIGE is a desktop app for manually coding gaze or behavior from video. It is designed for research work where one long video needs to be split into meaningful sections, labeled frame by frame, and exported as CSV data.

Screenshots and videos will be added later.

---

## English

### What MIGE Does

MIGE lets you open a video, mark important sections on the timeline, add names to those sections, and enter frame-by-frame labels with the keyboard.

You can use it for workflows like:

1. Open one long video.
2. Mark the parts you want to code.
3. Give each section a label, such as `pre_1`, `habi_1`, or `post_1`.
4. Code gaze or behavior frame by frame.
5. Export the data as CSV.
6. Export selected sections as separate video clips when needed.

### Main Features

- Video browser: shows the videos in the same folder, so you can move between videos easily.
- Video preview: shows the current frame and the active section name in the top-left corner.
- Waveform view: shows the audio waveform and lets you click or drag to move the playhead. When zoomed in, clicking the waveform moves the cursor to that precise point without immediately re-centering the view.
- Timeline sections: create highlighted sections, move them, resize them, and label them.
- Section labels: load preset section names from a JSON file such as `sections_settings.json`.
- Experiment CSV import: read PsychoPy-style CSV files and create timeline sections from event onset/offset times.
- Keyboard coding: press label keys to code the current frame.
- Section start/end shortcuts: press `1` to set the start point and `2` to set the end point.
- Section navigation shortcuts: press `Tab` to select the next section and `Shift+Tab` to select the previous section.
- Normal playback: press `Space` to play or pause the video.
- Auto section jump: during coding, MIGE can jump to the next section after finishing the current one.
- Preview effects: adjust brightness and contrast for easier viewing. These changes are for preview only.
- Settings window: edit label keys, modes, groups, section duration, section label JSON, and other shortcuts.
- CSV export: export frame-level labels.
- Calculated CSV export: export summaries by section, group, and mode.
- Section video export: export the selected section as a separate video file.

### Files MIGE Uses

MIGE keeps the main settings in:

```text
encode_settings.json
```

This file stores keyboard shortcuts, coding labels, groups, timing settings, and section settings.

Section label presets are stored in a separate JSON file, for example:

```text
sections_settings.json
sections_settings_2.json
```

These files can contain preset section names:

```json
{
  "default_section_labels": [
    {
      "section_ID": 1,
      "section_label": "pre_1",
      "group_ID": "test"
    },
    {
      "section_ID": 2,
      "section_label": "pre_2",
      "group_ID": "test"
    }
  ]
}
```

When a video is loaded, MIGE can apply these names to matching `section_ID`s. If the video already has different manual section names, MIGE will ask whether to apply the JSON names or keep the current manual names.

### Output Files

When you code a video, MIGE saves data next to the video file.

For a video called:

```text
sample.mp4
```

MIGE may create:

```text
sample_labels.csv
sample_sections.json
sample_calculated.csv
```

The regular CSV includes:

```text
frame,section_ID,section_label,group_ID,mode
```

The calculated CSV includes:

```text
section_ID,section_label,group_ID,mode,mode_count,total_frames,total_seconds,fps
```

### Install with uv

MIGE uses Python. The easiest way to install and run it is with `uv`.

Official uv installation guide: <https://docs.astral.sh/uv/getting-started/installation/>

#### 1. Install uv

macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installing uv, close and reopen your terminal.

#### 2. Open the MIGE folder

```bash
cd path/to/MIGE
```

#### 3. Install the app dependencies

```bash
uv sync
```

The first run may take a little time. uv can also install the needed Python version automatically.

#### 4. Start MIGE

```bash
uv run python main.py
```

### Optional: Install FFmpeg

MIGE can code videos without FFmpeg, but exporting selected sections as separate video clips needs FFmpeg.

macOS with Homebrew:

```bash
brew install ffmpeg
```

Windows users can install FFmpeg from the official FFmpeg website or with a package manager.

### Basic Use

1. Start MIGE with `uv run python main.py`.
2. Drag a video into the window, or click `+ Add Video`.
3. Use the video browser on the left to choose videos from the same folder.
4. Create section highlights on the timeline.
5. Add or edit section labels.
6. Use the keyboard to code frames.
7. Export the CSV when finished.

If your experiment software already saved stimulus timing, you can also use `Read Experiment CSV File` on the left side to create timeline sections automatically.

### Creating Sections

You can create sections in several ways:

- Drag on the timeline to create a highlighted section.
- Click `+ Add Section`.
- Press `1` to set a section start point.
- Press `2` to set a section end point.

If the playhead is inside an existing section, `1` or `2` updates that section. If the playhead is outside all sections, MIGE creates a new section.

New sections use the default duration from the settings. The default is 10 seconds.

### Navigating the Timeline

When the waveform is zoomed in, click or drag on the waveform to move the playhead to the exact visible point. The view stays where you clicked so you can make precise selections. After that, `Left` and `Right` still move to the previous or next frame and center the cursor in the zoomed timeline view.

Use `Tab` to select the next section in the section list. Use `Shift+Tab` to select the previous section. Selecting a section this way behaves like selecting it in the right-side section list: MIGE selects the section and jumps to its start frame.

### Importing Sections from an Experiment CSV

The left side has a `Read Experiment CSV File` area between the video browser and `Section Labels`.

Use it when a PsychoPy or experiment CSV already contains stimulus timing. The CSV should include at least these columns:

```text
phase,event,onset,offset,duration
```

When a CSV is loaded, MIGE opens a preview dialog.

- Rows where `phase` is `stimulus` are selected automatically.
- You can check or uncheck rows before importing.
- You can choose which columns to use for the section name, start time, and end time.
- By default, `event` is used as the section name/identifier, `onset` is used as the start time, and `offset` is used as the end time.
- The table uses a dark background with alternating text colors for readability.

CSV timing must be aligned to the video timeline. Before loading the CSV, move the playhead to the timeline point that should match CSV time 0, or edit `CSV time 0 in timeline` in the preview dialog.

After you confirm the preview, MIGE asks whether it is OK to replace the current timeline sections. If you continue, the imported rows replace the current sections. Existing frame-level coding stays on the same frames, but `section_ID` and `section_label` values are recalculated from the new sections.

### Moving Sections

You can move and resize highlighted sections on the timeline.

Important: moving a section does not move already coded frame data. The coded data stays on the original frames. This helps prevent accidental changes to completed coding.

If real-time section update is turned on, the highlight moves while you drag it. If it feels slow on a large video, turn off:

```text
enable_real_time_section_update
```

You can change this in the Settings window.

### Section Labels

The left side has a `Section Labels` area.

Use it to:

- see the labels loaded from the current section label JSON file
- change the JSON file
- apply labels to the current video

This is useful when different studies or tasks need different section names.

### Coding Frames

Default coding keys are:

```text
A = video_L
D = video_R
S = no
Q = see
W = no
```

Other default controls:

```text
Left / Right = previous / next frame
Up / Down    = previous / next video
Space        = play / pause
M            = switch coding mode
X            = fill an unlabeled gap
T            = switch frame/time display
1            = set section start
2            = set section end
Tab          = select next section
Shift+Tab    = select previous section
```

You can change these in the Settings window. The same shortcuts are stored in `encode_settings.json` under `app_keys`, where `tab` maps to `next_section` and `shift+tab` maps to `prev_section` by default.

### Exports

Use `Export CSV` for frame-level coding data.

Use `Export Calculated CSV` for summary data by section, group, and mode.

Use `Export Selected Video` to save the selected section as a separate video clip.

### Settings

Most settings can be changed from the GUI. You can also edit `encode_settings.json` directly if needed.

Common settings:

- label keys
- mode names
- group IDs
- app shortcuts
- default section duration
- section label JSON file
- real-time section update
- mouse wheel speed

---

## 日本語

### MIGEとは

MIGEは、動画を見ながら視線や行動を手動でコーディングするためのデスクトップアプリです。長い動画をそのまま読み込み、必要な部分だけをセクションとして指定し、フレームごとにラベルを入力できます。

たとえば、次のような流れで使えます。

1. 長い動画を開く。
2. コーディングしたい区間をタイムライン上で指定する。
3. 各セクションに `pre_1`、`habi_1`、`post_1` などの名前を付ける。
4. キーボードでフレームごとにラベルを入力する。
5. CSVとして書き出す。
6. 必要であれば、選択したセクションだけを別の動画として書き出す。

### 主な機能

- 動画ブラウザ: 同じフォルダ内の動画を左側に表示します。
- 動画プレビュー: 現在のフレームと、現在のセクション名を左上に表示します。
- 波形表示: 音声の波形を表示し、クリックやドラッグで再生位置を移動できます。拡大表示中は、クリックした波形上の正確な位置へカーソルを移動し、すぐには表示を中央へ戻しません。
- タイムラインのセクション: 区間を作成、移動、リサイズ、ラベル付けできます。
- セクションラベル: `sections_settings.json` などのJSONファイルから、セクション名を読み込めます。
- 実験CSV読み込み: PsychoPyなどのCSVに保存された刺激の開始・終了時刻から、タイムラインのセクションを自動作成できます。
- キーボード入力: キーを押して現在のフレームにラベルを付けます。
- セクション開始・終了ショートカット: `1` で開始位置、`2` で終了位置を設定します。
- セクション移動ショートカット: `Tab` で次のセクション、`Shift+Tab` で前のセクションを選択します。
- 通常再生: `Space` で動画を再生・停止できます。
- 自動セクション移動: コーディング中に現在のセクションが終わると、次のセクションへ移動できます。
- プレビュー調整: 明るさとコントラストを調整できます。これは見やすくするためだけで、書き出し動画には反映されません。
- 設定画面: ラベルキー、モード、グループ、セクション時間、セクションラベルJSON、ショートカットを編集できます。
- CSV書き出し: フレームごとのデータを書き出します。
- 集計CSV書き出し: セクション、グループ、モードごとの集計を書き出します。
- セクション動画書き出し: 選択したセクションだけを別動画として保存できます。

### MIGEが使うファイル

基本設定は次のファイルに保存されています。

```text
encode_settings.json
```

このファイルには、キーボードショートカット、ラベル、グループ、タイミング、セクション関連の設定が入っています。

セクション名のプリセットは、別のJSONファイルに保存できます。

```text
sections_settings.json
sections_settings_2.json
```

例:

```json
{
  "default_section_labels": [
    {
      "section_ID": 1,
      "section_label": "pre_1",
      "group_ID": "test"
    },
    {
      "section_ID": 2,
      "section_label": "pre_2",
      "group_ID": "test"
    }
  ]
}
```

動画を開いたとき、MIGEは `section_ID` に合うセクション名を自動で適用できます。すでに手動で付けた名前とJSONの名前が違う場合は、JSONの名前を使うか、今の手動入力を残すかを確認します。

### 出力されるファイル

たとえば動画名が次の場合:

```text
sample.mp4
```

MIGEは同じフォルダに次のようなファイルを作ります。

```text
sample_labels.csv
sample_sections.json
sample_calculated.csv
```

通常のCSVには次の列が入ります。

```text
frame,section_ID,section_label,group_ID,mode
```

集計CSVには次の列が入ります。

```text
section_ID,section_label,group_ID,mode,mode_count,total_frames,total_seconds,fps
```

### uvでインストールする方法

MIGEはPythonで動きます。インストールと実行には `uv` を使うのが簡単です。

uv公式インストールガイド: <https://docs.astral.sh/uv/getting-started/installation/>

#### 1. uvをインストールする

macOSまたはLinux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール後、ターミナルを一度閉じて開き直してください。

#### 2. MIGEのフォルダを開く

```bash
cd path/to/MIGE
```

#### 3. 必要なものをインストールする

```bash
uv sync
```

初回は少し時間がかかることがあります。必要なPythonがない場合、uvが自動で準備できます。

#### 4. MIGEを起動する

```bash
uv run python main.py
```

### 任意: FFmpegをインストールする

通常のコーディングだけならFFmpegは不要です。ただし、選択したセクションを動画として書き出すにはFFmpegが必要です。

macOSでHomebrewを使う場合:

```bash
brew install ffmpeg
```

Windowsの場合は、FFmpeg公式サイトまたはパッケージ管理ツールからインストールしてください。

### 基本的な使い方

1. `uv run python main.py` でMIGEを起動します。
2. 動画をウィンドウにドラッグするか、`+ Add Video` をクリックします。
3. 左側の動画ブラウザから、同じフォルダ内の動画を選べます。
4. タイムライン上にセクションを作ります。
5. セクション名を入力または編集します。
6. キーボードでフレームごとにラベルを入力します。
7. 作業が終わったらCSVを書き出します。

実験ソフトで刺激のタイミングを保存している場合は、左側の `Read Experiment CSV File` からタイムラインのセクションを自動作成できます。

### セクションを作る

セクションは次の方法で作れます。

- タイムライン上をドラッグする。
- `+ Add Section` をクリックする。
- `1` を押して開始位置を設定する。
- `2` を押して終了位置を設定する。

再生位置が既存のセクション内にある場合、`1` と `2` はそのセクションを編集します。再生位置がどのセクションにも入っていない場合は、新しいセクションを作ります。

新しいセクションの長さは設定で変更できます。初期設定は10秒です。

### タイムライン内を移動する

波形を拡大しているときは、波形上をクリックまたはドラッグすると、見えているその正確な位置へ再生位置を移動できます。クリック直後に表示は中央へ移動しないため、細かい位置を選びやすくなっています。その後で `Left` / `Right` を押すと、前後のフレームへ移動し、拡大タイムライン上でカーソルが中央に来ます。

`Tab` を押すと、右側のセクション一覧で次のセクションを選択した場合と同じように、次のセクションを選択して開始フレームへ移動します。`Shift+Tab` では前のセクションを選択して開始フレームへ移動します。

### 実験CSVからセクションを読み込む

左側の動画ブラウザと `Section Labels` の間に、`Read Experiment CSV File` があります。

PsychoPyなどの実験CSVに刺激のタイミングが保存されている場合、この機能でセクションを自動作成できます。CSVには少なくとも次の列が必要です。

```text
phase,event,onset,offset,duration
```

CSVを読み込むと、確認用のダイアログが開きます。

- `phase` が `stimulus` の行が自動で選択されます。
- 読み込む行は、チェックを付けたり外したりして変更できます。
- セクション名、開始時刻、終了時刻に使う列を選べます。
- 初期設定では、`event` をセクション名・識別名、`onset` を開始時刻、`offset` を終了時刻として使います。
- 表は見やすいように暗い背景と交互の文字色で表示されます。

CSVの時刻は、動画のタイムラインに合わせる必要があります。CSVを読み込む前に、CSVの0秒に対応する動画上の位置へ再生位置を移動するか、確認ダイアログの `CSV time 0 in timeline` を編集してください。

プレビューを確認したあと、現在のタイムラインセクションを置き換えてよいか確認されます。続行すると、読み込んだ行が現在のセクションを置き換えます。すでに入力したフレームごとのデータは同じフレームに残りますが、`section_ID` と `section_label` は新しいセクションに合わせて再計算されます。

### セクションを動かす

タイムライン上のセクションは、移動や長さの変更ができます。

重要: セクションを動かしても、すでに入力したフレームごとのデータは移動しません。入力済みデータは元のフレームに残ります。これにより、完了したコーディングが誤って変わるのを防げます。

リアルタイム更新がオンの場合、ドラッグ中にセクションの表示がその場で動きます。長い動画で重く感じる場合は、設定画面で次をオフにできます。

```text
enable_real_time_section_update
```

### セクションラベル

左側に `Section Labels` という欄があります。

ここでは次のことができます。

- 現在読み込んでいるセクションラベルを確認する。
- 使うJSONファイルを変更する。
- 現在の動画にセクションラベルを適用する。

研究やタスクごとにセクション名が違う場合に便利です。

### フレームをコーディングする

初期設定のラベルキーは次の通りです。

```text
A = video_L
D = video_R
S = no
Q = see
W = no
```

その他の初期ショートカット:

```text
Left / Right = 前のフレーム / 次のフレーム
Up / Down    = 前の動画 / 次の動画
Space        = 再生 / 停止
M            = コーディングモード切り替え
X            = ラベルなし区間の補完
T            = フレーム表示 / 時間表示の切り替え
1            = セクション開始位置を設定
2            = セクション終了位置を設定
Tab          = 次のセクションを選択
Shift+Tab    = 前のセクションを選択
```

これらは設定画面で変更できます。同じショートカットは `encode_settings.json` の `app_keys` にも保存されています。初期設定では、`tab` が `next_section`、`shift+tab` が `prev_section` に対応しています。

### 書き出し

`Export CSV` は、フレームごとのコーディングデータを書き出します。

`Export Calculated CSV` は、セクション、グループ、モードごとの集計を書き出します。

`Export Selected Video` は、選択したセクションだけを別の動画として保存します。

### 設定

多くの設定はGUIから変更できます。必要であれば、`encode_settings.json` を直接編集することもできます。

よく使う設定:

- ラベルキー
- モード名
- グループID
- アプリのショートカット
- 新しいセクションの初期時間
- セクションラベルJSONファイル
- セクションのリアルタイム更新
- マウスホイールの移動量
