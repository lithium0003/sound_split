# sound_split
sound spectrum analyzer

## description
A linux cuda implementation of [Fast recursive spectral analysis using variable window　(H.Nakatsuji)][ref1].

## screenshot
![screenshot](https://raw.githubusercontent.com/lithium0003/sound_split/docs/images/screen1.png "screenshot")

## compile
+ prepare cuda environment

```bash
sudo apt-get install build-essential
wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux
sudo bash cuda_9.2.148_396.37_linux
wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/patches/1/cuda_9.2.148.1_linux
sudo bash cuda_9.2.148.1_linux
```

and add this to .bashrc

```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

+ prepare libraries

```bash
sudo apt install libgtk-3-dev
sudo apt install libpulse-dev
```

+ clone source and compile

```bash
git clone https://github.com/lithium0003/sound_split.git
cd sound_split
```

edit cuda compute capability in Makefile

```Makefile
NVCCFLAGS := -O3 -I ~/NVIDIA_CUDA-9.2_Samples/common/inc/ -gencode=arch=compute_61,code=sm_61
```

and run make

```bash
make
```

## how to use
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

## License
These codes are licensed under CC0.

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png "CC0")](http://creativecommons.org/publicdomain/zero/1.0/deed.en)

## reference
[可変窓を用いた高速再帰的スペクトル解析　中辻 秀人　著][ref1]

[ref1]:https://www.amazon.co.jp/dp/4862238378
