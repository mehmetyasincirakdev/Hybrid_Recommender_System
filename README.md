# Hybrid Recommender System

MovieLens verisi üzerinde kullanıcı benzerliğine (user-based) ve film
benzerliğine (item-based) dayalı öneriler üretir. Uygulama; veri okuma,
öneri algoritmaları ve komut satırı arayüzü olarak ayrılmıştır ve sentetik
verilerle test edilebilir.

## Gereksinimler

- Python 3.10 veya üzeri
- MovieLens `movies.csv` ve `ratings.csv` dosyaları

Veri setini [GroupLens MovieLens](https://grouplens.org/datasets/movielens/)
sayfasından indirin. Arşivi açtıktan sonra dosyaları `data/raw/` dizinine
koyun. Beklenen sütunlar:

- `movies.csv`: `movieId`, `title` (diğer sütunlar yok sayılır)
- `ratings.csv`: `userId`, `movieId`, `rating`, `timestamp`

Film ve puan dosyaları dağıtım lisansı nedeniyle depoya eklenmez.

## Kurulum

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Öneri üretme

```powershell
hybrid-recommender --user-id 108170
```

Varsayılan dosya yolları `data/raw/movies.csv` ve `data/raw/ratings.csv`;
varsayılan olarak film başına en az 1000 puan aranır. Veri yolları ve eşik,
komut satırından değiştirilebilir:

```powershell
hybrid-recommender `
  --movies data/raw/movies.csv `
  --ratings data/raw/ratings.csv `
  --user-id 108170 `
  --min-movie-ratings 1000 `
  --count 5
```

Program kullanıcı tabanlı ve film tabanlı sonuçları ayrı ayrı gösterir.
Kullanıcı tabanlı yöntemde hedef kullanıcının izlediklerinin en az %60'ını
izleyen ve korelasyonu en az `0.65` olan kullanıcılar kullanılır. Film tabanlı
yöntem, kullanıcının zaman damgasına göre en son verdiği 5 yıldızlı filmden
hareket eder; iki yöntem de daha önce puanlanan filmleri önermez. Uygun
öneri bulunmazsa boş sonuç gösterilir. Kullanıcı ID'si filtrelenmiş matriste
yoksa program anlaşılır bir hata mesajıyla durur.

## Geliştirme ve test

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Proje yapısı:

```text
src/hybrid_recommender/
  data.py          # CSV doğrulama ve kullanıcı-film matrisi
  recommenders.py  # User-based ve item-based algoritmalar
  cli.py           # Komut satırı arayüzü
tests/
  test_data.py
  test_recommenders.py
data/raw/          # Yerel MovieLens dosyaları (git dışında)
```
