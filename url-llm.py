import streamlit as st
from langchain_community.chat_models import ChatGooglePalm
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

st.title("News Research tool 📈")

main_placeholder = st.empty()
api_key = "AIzaSyD4k8k0c3LDwcGmu1z8fbxQ8TOWxYU__-Q"

llm = ChatGooglePalm(google_api_key=api_key, temperature=0.7, max_tokens=500)
vector_file_path = "vector_index-1"

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb = FAISS.load_local(vector_file_path, embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(score_threshold=0.7)

customer_profile = {
    "CustomerID": st.text_input("Customer ID", "12345"),
    "Age": st.number_input("Age", 30),
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "Job": st.text_input("Job", "Software Engineer"),
    "Education": st.text_input("Education", "Bachelor's Degree"),
    "MaritalStatus": st.selectbox("Marital Status", ["Single", "Married"]),
    "Balance": st.number_input("Balance", 20000),
    "Tenure": st.number_input("Tenure (years)", 5),
    "CreditScore": st.number_input("Credit Score", 750),
    "HasLoan": st.selectbox("Has Loan", ["Yes", "No"]),
    "TypeOfLoan": st.text_input("Type of Loan", "Personal Loan"),
    "AvgTransactionPerMonth": st.number_input("Avg Transactions Per Month", 20),
    "HasCrCard": st.selectbox("Has Credit Card", ["Yes", "No"]),
    "EstimatedSalary": st.number_input("Estimated Salary", 80000),
    "NumOfProducts": st.number_input("Number of Products", 3),
    "UseOfDigitalBankServices": st.selectbox("Use of Digital Bank Services", ["Low", "Moderate", "High"]),
    "KnowledgeAboutServices": st.selectbox("Knowledge About Services", ["Low", "Moderate", "High"]),
    "InvestmentInBankSchemes": st.selectbox("Investment in Bank Schemes", ["Yes", "No"]),
    "IsActiveMember": st.selectbox("Is Active Member", ["Yes", "No"]),
    "Rating": st.number_input("Rating", 4),
    "ServicesUsed": st.text_input("Services Used", "Savings Account, Credit Card, Personal Loan"),
    "ReviewSentiment": st.text_input("Review Sentiment", "Positive"),
    "KeywordsInComplaint": st.text_input("Keywords in Complaint", "None"),
    "IssueResolved": st.text_input("Issue Resolved", "N/A"),
    "CustomerLeft": st.selectbox("Customer Left", ["Yes", "No"]),
    "App": st.selectbox("App Usage", ["Frequent", "Rare"]),
    "ServiceOffline": st.selectbox("Service Offline Usage", ["Frequent", "Rare"]),
    "EaseOfAccessOfBankingServiceOnline": st.selectbox("Ease of Access of Banking Service Online",
                                                       ["Low", "Moderate", "High"]),
    "PaymentDeclineServerIssue": st.selectbox("Payment Decline Server Issue", ["Yes", "No"]),
    "FraudSecurity": st.selectbox("Fraud Security", ["Low", "Moderate", "High"])
}

context = """
The bank aims to improve customer retention by providing personalized services and products, enhancing customer support and engagement, and optimizing both digital and offline experiences. The bank offers various financial services, including savings accounts, credit cards, loans, and investment schemes. 

Challenges include:
1. High competition in the financial services sector.
2. Increasing customer expectations for personalized and seamless banking experiences.
3. Managing and resolving customer complaints and issues promptly.
4. Ensuring robust security measures to protect against fraud and security concerns.
5. Encouraging customers to use digital banking services effectively.

Goals:
1. Increase customer satisfaction and loyalty.
2. Reduce customer churn rate.
3. Enhance customer engagement through tailored communication and services.
4. Improve the overall customer experience across all banking channels.
5. Implement effective loyalty programs and rewards to incentivize long-term relationships.

With this context, please consider the following attributes to provide personalized strategies for customer retention:

Customer Attributes:
- CustomerID: Unique identifier for the customer.
- Age: Age of the customer.
- Gender: Gender of the customer.
- Job: Customer's occupation.
- Education: Customer's educational background.
- MaritalStatus: Marital status of the customer.
- Balance: Account balance of the customer.
- Tenure: Number of years the customer has been with the bank.
- CreditScore: Customer's credit score.
- HasLoan: Whether the customer has a loan.
- TypeOfLoan: Type of loan the customer has.
- AvgTransactionPerMonth: Average number of transactions per month.
- HasCrCard: Whether the customer has a credit card.
- EstimatedSalary: Estimated annual salary of the customer.
- NumOfProducts: Number of products the customer has with the bank.
- UseOfDigitalBankServices: Customer's usage level of digital banking services.
- KnowledgeAboutServices: Customer's knowledge about bank services.
- InvestmentInBankSchemes: Whether the customer has investments in bank schemes.
- IsActiveMember: Whether the customer is an active member.
- Rating: Customer's rating of the bank's services.
- ServicesUsed: Services the customer uses.
- ReviewSentiment: Sentiment of the customer's reviews.
- KeywordsInComplaint: Keywords found in customer complaints.
- IssueResolved: Whether the customer's issues have been resolved.
- CustomerLeft: Whether the customer has left the bank.
- App: Usage of the bank's mobile app.
- ServiceOffline: Usage of offline banking services.
- EaseOfAccessOfBankingServiceOnline: Ease of access to online banking services.
- PaymentDeclineServerIssue: Whether the customer faced payment declines due to server issues.
- FraudSecurity: Customer's concerns about fraud and security.
"""

prompt_template = """
{context}
Based on the customer profile provided, suggest strategies to improve customer retention for the following customer attributes:

CustomerID: {CustomerID}
Age: {Age}
Gender: {Gender}
Job: {Job}
Education: {Education}
MaritalStatus: {MaritalStatus}
Balance: {Balance}
Tenure: {Tenure}
CreditScore: {CreditScore}
HasLoan: {HasLoan}
TypeOfLoan: {TypeOfLoan}
AvgTransactionPerMonth: {AvgTransactionPerMonth}
HasCrCard: {HasCrCard}
EstimatedSalary: {EstimatedSalary}
NumOfProducts: {NumOfProducts}
UseOfDigitalBankServices: {UseOfDigitalBankServices}
KnowledgeAboutServices: {KnowledgeAboutServices}
InvestmentInBankSchemes: {InvestmentInBankSchemes}
IsActiveMember: {IsActiveMember}
Rating: {Rating}
ServicesUsed: {ServicesUsed}
ReviewSentiment: {ReviewSentiment}
KeywordsInComplaint: {KeywordsInComplaint}
IssueResolved: {IssueResolved}
CustomerLeft: {CustomerLeft}
App: {App}
ServiceOffline: {ServiceOffline}
EaseOfAccessOfBankingServiceOnline: {EaseOfAccessOfBankingServiceOnline}
PaymentDeclineServerIssue: {PaymentDeclineServerIssue}
FraudSecurity: {FraudSecurity}
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=[
        "context", "CustomerID", "Age", "Gender", "Job", "Education",
        "MaritalStatus", "Balance", "Tenure", "CreditScore",
        "HasLoan", "TypeOfLoan", "AvgTransactionPerMonth", "HasCrCard",
        "EstimatedSalary", "NumOfProducts", "UseOfDigitalBankServices",
        "KnowledgeAboutServices", "InvestmentInBankSchemes", "IsActiveMember",
        "Rating", "ServicesUsed", "ReviewSentiment", "KeywordsInComplaint",
        "IssueResolved", "CustomerLeft", "App", "ServiceOffline",
        "EaseOfAccessOfBankingServiceOnline", "PaymentDeclineServerIssue",
        "FraudSecurity"
    ]
)

chain = LLMChain(llm=llm, prompt=PROMPT, output_key="Strats")
if st.button("Generate Strategies"):
    try:
        formatted_prompt = PROMPT.format(
            context=context,
            **customer_profile
        )
        result = chain.run({
            "context": context,
            "CustomerID": customer_profile["CustomerID"],
            "Age": customer_profile["Age"],
            "Gender": customer_profile["Gender"],
            "Job": customer_profile["Job"],
            "Education": customer_profile["Education"],
            "MaritalStatus": customer_profile["MaritalStatus"],
            "Balance": customer_profile["Balance"],
            "Tenure": customer_profile["Tenure"],
            "CreditScore": customer_profile["CreditScore"],
            "HasLoan": customer_profile["HasLoan"],
            "TypeOfLoan": customer_profile["TypeOfLoan"],
            "AvgTransactionPerMonth": customer_profile["AvgTransactionPerMonth"],
            "HasCrCard": customer_profile["HasCrCard"],
            "EstimatedSalary": customer_profile["EstimatedSalary"],
            "NumOfProducts": customer_profile["NumOfProducts"],
            "UseOfDigitalBankServices": customer_profile["UseOfDigitalBankServices"],
            "KnowledgeAboutServices": customer_profile["KnowledgeAboutServices"],
            "InvestmentInBankSchemes": customer_profile["InvestmentInBankSchemes"],
            "IsActiveMember": customer_profile["IsActiveMember"],
            "Rating": customer_profile["Rating"],
            "ServicesUsed": customer_profile["ServicesUsed"],
            "ReviewSentiment": customer_profile["ReviewSentiment"],
            "KeywordsInComplaint": customer_profile["KeywordsInComplaint"],
            "IssueResolved": customer_profile["IssueResolved"],
            "CustomerLeft": customer_profile["CustomerLeft"],
            "App": customer_profile["App"],
            "ServiceOffline": customer_profile["ServiceOffline"],
            "EaseOfAccessOfBankingServiceOnline": customer_profile["EaseOfAccessOfBankingServiceOnline"],
            "PaymentDeclineServerIssue": customer_profile["PaymentDeclineServerIssue"],
            "FraudSecurity": customer_profile["FraudSecurity"]
        })
        st.write(result)
    except Exception as e:
        st.error(f"An error occurred: {e}")

