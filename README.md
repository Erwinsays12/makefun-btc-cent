# Peramal harga Bitcoin 4 jam ke depan (btc-cent-forecast-01)

Program ini membaca harga penutupan Bitcoin per menit selama 72 jam terakhir
sampai tiap checkpoint, lalu menebak harga 240 menit (4 jam) setelah checkpoint.

## Cara kerja

Tebakan = harga penutupan terakhir × exp(0,5 × revert + 0,5 × ridge):

- **revert** — mean reversion ke rata-rata 24 jam: `-0,3 × (log P − rata-rata
  log P 1440 menit terakhir)`.
- **ridge** — regresi ridge (lambda 10) yang memprediksi log-return 240 menit
  dari 8 fitur kausal (return 15/60/240/720/1440 menit, selisih EMA(15)−EMA(120),
  deviasi dari rata-rata 24 jam, volatilitas per menit). Regresi dilatih di dalam
  tiap run, hanya dari histori kasus itu sendiri (pseudo-checkpoint tiap 60 menit
  dari indeks 1500 sampai t−240), jadi tidak ada kebocoran masa depan.

Model dipilih lewat backtest walk-forward atas enam kasus practice di kit
(516 pseudo-checkpoint): MAE 38.842 sen vs 43.433 sen untuk baseline persistence,
lebih baik di semua kasus practice dan di kedua paruh jendela. Pada lima kasus
practice yang actual-nya diketahui, skornya 36.726 sen vs 47.055 sen persistence.
(Baseline practice resmi 6 kasus: 44.521,58 sen.)

## Menjalankan

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Membaca setiap `<id>.in` di direktori input dan menulis `<id>.out` berisi
`{"forecasts":[{"case":"<id>","price":<tebakan>}]}`. Deterministik (tanpa
random, jam, atau jaringan), standar library saja, ~0,05 detik per kasus.
