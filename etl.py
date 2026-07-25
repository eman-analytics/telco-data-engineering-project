import pandas as pd

# قراءة البيانات
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Rows before cleaning:", len(df))

# إزالة الفراغات
df.columns = df.columns.str.strip()

# تحويل TotalCharges إلى رقم
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# حذف الصفوف التي فيها قيم مفقودة
df = df.dropna()

print("Rows after cleaning:", len(df))

print(df.dtypes)

print(df.head())
from sqlalchemy import create_engine

# بيانات الاتصال بقاعدة البيانات
user = "postgres"
password = "0541"
host = "localhost"
port = "5432"
database = "telco_churn"

# إنشاء الاتصال
engine = create_engine(
    f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
)

# تحميل البيانات إلى PostgreSQL
df.to_sql(
    "customers",
    engine,
    if_exists="replace",
    index=False
)

print("Data loaded successfully into PostgreSQL!")
df.to_sql(
    "customer_churn",
    engine,
    if_exists="replace",
    index=False
)

print("Table created successfully!")