
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.

#############################################
# Görev 1: Verinin Hazırlanması
#############################################

import pandas as pandas
pandas.pandas.set_option('display.max_columns', None)
pandas.pandas.set_option('display.width', 300)


# Adım 1: Movie ve Rating veri setlerini okutunuz.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti
movie = pandas.read_csv('Modül_4_Tavsiye_Sistemleri/datasets/movie.csv')
movie.head()
movie.shape

# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti
rating = pandas.read_csv('Modül_4_Tavsiye_Sistemleri/datasets/rating.csv')
rating.head()
rating.shape
rating["userId"].nunique()


# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.
dataframe = movie.merge(rating, how="left", on="movieId")
dataframe.head()
dataframe.shape


# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.
comment_counts = pandas.DataFrame(dataframe["title"].value_counts())
comment_counts

# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = dataframe[~dataframe["title"].isin(rare_movies)]
common_movies.shape


# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.

user_movie_dataframe = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_dataframe.head()


# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım
def create_user_movie_dataframe():
    import pandas as pandas
    movie = pandas.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pandas.read_csv('datasets/movie_lens_dataset/rating.csv')
    dataframe = movie.merge(rating, how="left", on="movieId")
    comment_counts = pandas.DataFrame(dataframe["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = dataframe[~dataframe["title"].isin(rare_movies)]
    user_movie_dataframe = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_dataframe

user_movie_dataframe = create_user_movie_dataframe()


#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.
random_user = 108170

# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_dataframe adında yeni bir dataframe oluşturunuz.
random_user_dataframe = user_movie_dataframe[user_movie_dataframe.index == random_user]
random_user_dataframe.head()
random_user_dataframe.shape

# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.
movies_watched = random_user_dataframe.columns[random_user_dataframe.notna().any()].to_list()
movies_watched

movie.columns[movie.notna().any()].to_list()

#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_dataframe'ten seçiniz ve movies_watched_dataframe adında yeni bir dataframe oluşturuyoruz.
movies_watched_dataframe = user_movie_dataframe[movies_watched]
movies_watched_dataframe.head()
movies_watched_dataframe.shape

# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
# Ve yeni bir dataframe oluşturuyoruz.
user_movie_count = movies_watched_dataframe.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head(5)

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
len(users_same_movies)



#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_dataframe dataframe’ini filtreleyiniz.
final_dataframe = movies_watched_dataframe[movies_watched_dataframe.index.isin(users_same_movies)]
final_dataframe.head()
final_dataframe.shape

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_dataframe dataframe’i oluşturunuz.
corr_dataframe = final_dataframe.T.corr().unstack().sort_values()
corr_dataframe = pandas.DataFrame(corr_dataframe, columns=["corr"])
corr_dataframe.index.names = ['user_id_1', 'user_id_2']
corr_dataframe = corr_dataframe.reset_index()

#corr_dataframe[corr_dataframe["user_id_1"] == random_user]



# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
top_users = corr_dataframe[(corr_dataframe["user_id_1"] == random_user) & (corr_dataframe["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape
top_users.head()

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].unique()
top_users_ratings.head()



#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

# Adım 2: Film id’sive her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_dataframe adında yeni bir
# dataframe oluşturunuz.
recommendation_dataframe = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_dataframe = recommendation_dataframe.reset_index()
recommendation_dataframe.head()

# Adım 3: Adım3: recommendation_dataframe içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.
recommendation_dataframe[recommendation_dataframe["weighted_rating"] > 3.5]
movies_to_be_recommend = recommendation_dataframe[recommendation_dataframe["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)


# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]

# 0    Mystery Science Theater 3000: The Movie (1996)
# 1                               Natural, The (1984)
# 2                             Super Troopers (2001)
# 3                         Christmas Story, A (1983)
# 4                       Sonatine (Sonachine) (1993)



#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 108170

# Adım 1: movie,rating veri setlerini okutunuz.
movie = pandas.read_csv('Modül_4_Tavsiye_Sistemleri/datasets/movie.csv')
rating = pandas.read_csv('datasets/movie_lens_dataset/rating.csv')

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_dataframe dataframe’ini seçilen film id’sine göre filtreleyiniz.
movie[movie["movieId"] == movie_id]["title"].values[0]
movie_dataframe = user_movie_dataframe[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_dataframe

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
user_movie_dataframe.corrwith(movie_dataframe).sort_values(ascending=False).head(10)

# Son iki adımı uygulayan fonksiyon
def item_based_recommender(movie_name, user_movie_dataframe):
    movie = user_movie_dataframe[movie_name]
    return user_movie_dataframe.corrwith(movie).sort_values(ascending=False).head(10)

# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_dataframe)
# 1'den 6'ya kadar. 0'da filmin kendisi var. Onu dışarda bıraktık.
movies_from_item_based[1:6].index


# 'My Science Project (1985)',
# 'Mediterraneo (1991)',
# 'Old Man and the Sea,
# The (1958)',
# 'National Lampoon's Senior Trip (1995)',
# 'Clockwatchers (1997)']



