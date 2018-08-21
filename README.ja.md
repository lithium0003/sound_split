# sound_split
音声スペクトル解析プログラム

## 説明
[可変窓を用いた高速再帰スペクトル解析　中辻 秀人　著][ref1]で解説されているアルゴリズムのcuda実装及び、LinuxのGUIプログラム
(著者とは無関係です。)

## スクリーンショット
![screenshot](https://raw.githubusercontent.com/lithium0003/sound_split/docs/images/screen1.png "screenshot")

## コンパイル方法
+ cuda環境のインストール

```bash
sudo apt-get install build-essential
wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux
sudo bash cuda_9.2.148_396.37_linux
wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/patches/1/cuda_9.2.148.1_linux
sudo bash cuda_9.2.148.1_linux
```

.bashrcに環境変数を追記

```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

+ 依存パッケージの追加

```bash
sudo apt install libgtk-3-dev
sudo apt install libpulse-dev
```

+ コンパイル

```bash
git clone https://github.com/lithium0003/sound_split.git
cd sound_split
```

Makefileの以下の行を、cudaのcompute capabilityに合わせて変更

```Makefile
NVCCFLAGS := -O3 -I ~/NVIDIA_CUDA-9.2_Samples/common/inc/ -gencode=arch=compute_61,code=sm_61
```

makeを実行

```bash
make
```

## 使い方
```shell-session
$ ./sound_split [optinos] input.wav
options
  --mono: force load as mono
  --start [time]: skip first [time] samples in wav file
  --length [time]: analyze only [time] length
  --filter: filter mode
time option
  10.05 -> 10.05sec
  1min20sec or 1h10m5.02sec will be ok
```

引数に、解析したいwavファイルを与えます。


## 免責
間違いの無いように注意しましたが、参考文献から移植上のミスがある場合があります。
重要なものに利用される場合は、ご自身で間違いの無いことを確認ください。

## License
このコード自体はCC0で公開します。
These codes are licensed under CC0.

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png "CC0")](http://creativecommons.org/publicdomain/zero/1.0/deed.en)

アルゴリズムに関して、著者から別途の指示がある場合はそれに従ってください。その他特許等のライセンスに関しては関知しません。

## 参考文献
[可変窓を用いた高速再帰スペクトル解析　中辻 秀人　著][ref1]

[ref1]:https://www.amazon.co.jp/dp/4862238378
