# Hybrid Recommender System

MovieLens verisi üzerinde kullanıcı benzerliğine (user-based) ve film
benzerliğine (item-based) dayalı öneriler üretir. Veri yükleme, algoritmalar ve
komut satırı arayüzü birbirinden ayrıdır; sentetik verilerle test edilebilir.

## Gereksinimler ve veri

- Python 3.10 veya üzeri
- MovieLens `movies.csv` ve `ratings.csv`

Veri setini [GroupLens MovieLens](https://grouplens.org/datasets/movielens/)
sayfasından indirin. Arşivi açtıktan sonra CSV'leri `data/raw/` dizinine
koyun. Beklenen sütunlar:

- `movies.csv`: `movieId`, `title`
- `ratings.csv`: `userId`, `movieId`, `rating`, `timestamp`

Ek sütunlar (ör. `genres`) yok sayılır. Veri dosyaları depoya eklenmez.

## Kurulum

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Kullanım

```powershell
hybrid-recommender --user-id 108170
```

Varsayılan veri yolları `data/raw/movies.csv` ve `data/raw/ratings.csv`;
varsayılan film popülerlik eşiği film başına 1000 puandır. Örnek:

```powershell
hybrid-recommender `
  --movies data/raw/movies.csv `
  --ratings data/raw/ratings.csv `
  --user-id 108170 `
  --min-movie-ratings 1000 `
  --count 5
```

User-based öneriler, hedef kullanıcının izlediklerinin en az %60'ını izleyen ve
Pearson korelasyonu en az `0.65` olan kullanıcıları kullanır. Item-based
yöntem, kullanıcının matriste bulunan en son 5 yıldızlı filminden hareket eder.
İki yöntem de kullanıcının daha önce puanladığı filmleri önermez. Uygun
öneri bulunmazsa boş sonuç gösterilir; hedef kullanıcı filtrelenmiş matriste
yoksa anlaşılır bir hata mesajı verilir.

Modül olarak da çalıştırılabilir:

```powershell
python -m hybrid_recommender --user-id 108170
```

## Test ve kalite kontrolleri

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## Proje yapısı

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
