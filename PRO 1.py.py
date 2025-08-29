import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# بارگذاری فایل CSV در DataFrame
df = pd.read_csv("CAR.CSV")

# نمایش ابعاد داده (تعداد سطر و ستون)
print("Shape of data:", df.shape)

# نمایش چند سطر ابتدایی از داده‌ها
print(df.head())

# بررسی مقادیر گم‌شده
print("Missing values:\n", df.isnull().sum())
# فرض کنید ستون‌هایی از نوع object یا category وجود دارد
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# تبدیل ستون‌های دسته‌ای به یک‌داغ (dummy variables)
df_encoded = pd.get_dummies(df, columns=categorical_cols)
if 'GRADE' in df_encoded.columns:
    X = df_encoded.drop('GRADE', axis=1)
    y = df_encoded['GRADE']
else:
    X = df_encoded.copy()
    y = None

print("Features shape:", X.shape)
if y is not None:
    print("Labels shape:", y.shape)
# اعمال KMeans با k=3
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
clusters = kmeans.predict(X)

# کاهش ابعاد به 2 بعد با PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# نمودار پراکندگی با رنگ‌بندی بر اساس برچسب‌های خوشه
plt.figure(figsize=(6,5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1],
            c='red', s=100, marker='X', label='Centers')
plt.title("koshebandi ba k-means va kahesh ba PCA")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()
plt.show()
if y is not None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ایجاد مدل KNN با k=5 (به عنوان مثال)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # پیش‌بینی روی داده‌های تست
    y_pred = knn.predict(X_test)

    print("Predictions (first 10):", y_pred[:10])
    if y is not None:
        acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    # ماتریس سردرگمی
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # نمودار ماتریس سردرگمی
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
    disp.plot(cmap='viridis')
    plt.title("matric sardargomi")
    plt.show()

    # گزارش طبقه‌بندی
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)
