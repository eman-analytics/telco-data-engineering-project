1. Total Customers
SELECT COUNT(*) AS total_customers
FROM customer_churn;

2. Churn Distribution
SELECT "Churn", COUNT(*) AS customers
FROM customer_churn
GROUP BY "Churn";

 3. Average Monthly Charges by Contract
SELECT "Contract",
       AVG("MonthlyCharges") AS avg_monthly_charges
FROM customer_churn
GROUP BY "Contract";

4. Average Tenure by Churn
SELECT "Churn",
       AVG("tenure") AS avg_tenure
FROM customer_churn
GROUP BY "Churn";

5. Customers by Internet Service
SELECT "InternetService",
       COUNT(*) AS customers
FROM customer_churn
GROUP BY "InternetService";
