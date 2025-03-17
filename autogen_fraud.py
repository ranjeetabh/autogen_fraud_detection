## This script is a comprehensive implementation of a fraud detection system and a customer service chatbot for a financial organization.


## Import necessary libraries
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


# Load environment variables
# This is used to store sensitive information like API keys 
load_dotenv()

# Configure API keys
config_list = [
    {
        "model": "", ##Enter model name of your choice
        "api_key": os.environ.get("API_KEY"),
        "api_base": "<base URL>",
        "api_type": "<api type>"
    }
]

# Configuration for the agents
llm_config = {
    "config_list": config_list,
    "cache_seed": 42,
    "temperature": 0.2,
    "timeout": 120,
}

# Simulated transaction data generator
def generate_transaction_data(num_transactions=1000, fraud_rate=0.05):
    """Generate simulated banking transaction data with some fraudulent transactions."""
    
    # Create random transaction data
    np.random.seed(42)
    transactions = []
    
    # Generate timestamps for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for i in range(num_transactions):
        # Generate random transaction amounts with different distributions
        amount = np.random.lognormal(mean=4.5, sigma=1.2)  # Most transactions between $10-$200
        
        # Generate transaction timestamp
        random_seconds = np.random.randint(0, int((end_date - start_date).total_seconds()))
        timestamp = start_date + timedelta(seconds=random_seconds)
        
        # Assign transaction category
        categories = ["groceries", "dining", "entertainment", "utilities", "travel", "shopping", "transfer"]
        category = np.random.choice(categories, p=[0.25, 0.2, 0.1, 0.15, 0.1, 0.15, 0.05])
        
        # Generate merchant name based on category
        merchants = {
            "groceries": ["Walmart", "Kroger", "Safeway", "Whole Foods", "Trader Joe's"],
            "dining": ["McDonald's", "Starbucks", "Chipotle", "Local Restaurant", "Food Delivery"],
            "entertainment": ["Netflix", "Movie Theater", "Concert Tickets", "Spotify"],
            "utilities": ["Electric Company", "Water Services", "Internet Provider", "Gas Company"],
            "travel": ["Airline Tickets", "Hotel Stay", "Car Rental", "Travel Agency"],
            "shopping": ["Amazon", "Target", "Best Buy", "Clothing Store", "Department Store"],
            "transfer": ["Bank Transfer", "Venmo", "PayPal", "Zelle"]
        }
        merchant = np.random.choice(merchants[category])
        
        # Generate location (city)
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego"]
        location = np.random.choice(cities)
        
        # Generate account information
        account_id = f"ACCT{np.random.randint(10000, 99999)}"
        customer_id = f"CUST{np.random.randint(1000, 9999)}"
        
        # Determine if transaction is fraudulent (5% chance)
        is_fraud = np.random.random() < fraud_rate
        
        # If fraudulent, modify some parameters to make it anomalous
        if is_fraud:
            # Fraudulent transactions may have unusual amounts
            amount = np.random.choice([
                np.random.lognormal(mean=8, sigma=1),  # Unusually large amount
                np.random.lognormal(mean=2, sigma=0.5)  # Many small transactions
            ])
            
            # Fraudulent transactions may occur at unusual times
            hour = np.random.randint(1, 5)  # Between 1 AM and 5 AM
            timestamp = timestamp.replace(hour=hour)
            
            # Fraudulent transactions may be in unusual locations
            unusual_locations = ["Foreign Country", "Unusual City", "Known Fraud Hotspot"]
            location = np.random.choice(unusual_locations)
        
        # Create transaction
        transaction = {
            "transaction_id": f"TX{i+1000}",
            "account_id": account_id,
            "customer_id": customer_id,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": round(amount, 2),
            "category": category,
            "merchant": merchant,
            "location": location,
            "is_fraud": is_fraud
        }
        
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)

# Generate the transaction data
df_transactions = generate_transaction_data(num_transactions=1000, fraud_rate=0.05)

# Save transactions to a CSV file for use by the agents
df_transactions.to_csv("banking_transactions.csv", index=False)

# Create a class to implement the fraud detection system
class FraudDetectionSystem:
    def __init__(self, transaction_data):
        self.transaction_data = transaction_data
        self.model = None
        self.fraud_scores = None
    
    def preprocess(self):
        """Preprocess the transaction data for anomaly detection."""
        # Convert timestamp to datetime
        self.transaction_data['timestamp'] = pd.to_datetime(self.transaction_data['timestamp'])
        
        # Extract features from timestamp
        self.transaction_data['hour'] = self.transaction_data['timestamp'].dt.hour
        self.transaction_data['day_of_week'] = self.transaction_data['timestamp'].dt.dayofweek
        
        # Create feature matrix
        # One-hot encode categorical variables
        features = pd.get_dummies(self.transaction_data[['category', 'location']], drop_first=True)
        
        # Add numerical features
        features['amount'] = self.transaction_data['amount']
        features['hour'] = self.transaction_data['hour']
        features['day_of_week'] = self.transaction_data['day_of_week']
        
        return features
    
    def train_model(self):
        """Train an anomaly detection model using Isolation Forest."""
        features = self.preprocess()
        
        # Initialize and train Isolation Forest model
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.model.fit(features)
        
        # Generate anomaly scores
        self.fraud_scores = self.model.decision_function(features)
        
        # Add scores to the transaction data
        self.transaction_data['anomaly_score'] = self.fraud_scores
        
        # Predict anomalies (1 for normal, -1 for anomalies)
        anomalies = self.model.predict(features)
        self.transaction_data['predicted_fraud'] = anomalies == -1
        
        # Calculate accuracy
        actual_fraud = self.transaction_data['is_fraud']
        predicted_fraud = self.transaction_data['predicted_fraud']
        accuracy = (actual_fraud == predicted_fraud).mean()
        
        return accuracy
    
    def get_fraud_alerts(self, threshold=-0.2):
        """Get transactions that are likely fraudulent based on the anomaly score."""
        if self.fraud_scores is None:
            raise ValueError("Model has not been trained yet.")
        
        fraud_alerts = self.transaction_data[self.transaction_data['anomaly_score'] < threshold].copy()
        return fraud_alerts[['transaction_id', 'account_id', 'customer_id', 'timestamp', 
                            'amount', 'category', 'merchant', 'location', 'anomaly_score', 
                            'is_fraud', 'predicted_fraud']]
    
    def visualize_anomalies(self):
        """Create visualization of the anomalies in the dataset."""
        plt.figure(figsize=(12, 6))
        
        # Plot transaction amounts
        plt.subplot(1, 2, 1)
        plt.scatter(
            range(len(self.transaction_data)), 
            self.transaction_data['amount'],
            c=self.transaction_data['predicted_fraud'].map({True: 'red', False: 'blue'}),
            alpha=0.5
        )
        plt.title('Transaction Amounts with Detected Anomalies')
        plt.xlabel('Transaction Index')
        plt.ylabel('Amount ($)')
        plt.colorbar(label='Predicted Fraud')
        
        # Plot anomaly scores
        plt.subplot(1, 2, 2)
        plt.hist(self.fraud_scores, bins=50)
        plt.title('Distribution of Anomaly Scores')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('fraud_detection_visualization.png')
        
        return 'fraud_detection_visualization.png'

# Create a customer service knowledge base
customer_service_kb = [
    {
        "question": "How do I report a lost or stolen card?",
        "answer": "To report a lost or stolen card, please call our 24/7 customer service line at 1-800-555-1234 immediately. You can also temporarily freeze your card through the mobile app or online banking."
    },
    {
        "question": "What are the fees for international transactions?",
        "answer": "Our bank charges a 3% fee for international transactions. This includes purchases made in foreign currencies or transactions processed outside the United States."
    },
    {
        "question": "How do I set up direct deposit?",
        "answer": "To set up direct deposit, provide your employer with your account number and our routing number (021000021). You can find these details on your checks, through online banking, or by contacting customer service."
    },
    {
        "question": "What should I do if I notice a fraudulent transaction?",
        "answer": "If you notice a fraudulent transaction, please contact our fraud department immediately at 1-800-555-5678. We recommend also changing your online banking password and monitoring your account for any additional suspicious activity."
    },
    {
        "question": "How do I apply for a loan?",
        "answer": "You can apply for a loan through our online banking portal, mobile app, or by visiting any branch location. To speed up the process, please have your financial information ready, including income verification and credit history."
    }
]

# Create a class to implement the customer service chatbot
# Provides answers to common customer service questions
class CustomerServiceChatbot:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
    
    def save_knowledge_base(self, filename="customer_service_kb.json"):
        """Save the knowledge base to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)
    
    def load_knowledge_base(self, filename="customer_service_kb.json"):
        """Load the knowledge base from a JSON file."""
        with open(filename, 'r') as f:
            self.knowledge_base = json.load(f)
    
    def get_answer(self, question):
        """Simple implementation to get an answer from the knowledge base."""
        for item in self.knowledge_base:
            if question.lower() in item["question"].lower():
                return item["answer"]
        
        return "I'm sorry, I don't have information on that topic. Please contact our customer service at 1-800-555-1234 for assistance."

# Initialize the customer service chatbot
customer_chatbot = CustomerServiceChatbot(customer_service_kb)
customer_chatbot.save_knowledge_base()

# Initialize the fraud detection system
fraud_system = FraudDetectionSystem(df_transactions)
accuracy = fraud_system.train_model()
fraud_alerts = fraud_system.get_fraud_alerts()
visualization_path = fraud_system.visualize_anomalies()

# Create AutoGen agents for the financial organization
# Create an agent that acts as a fraud detection expert
fraud_expert = autogen.AssistantAgent(
    name="Fraud_Detection_Expert",
    system_message="""You are an expert in fraud detection for financial organizations. 
    You can analyze transaction data to identify potential fraudulent activities and provide insights on how to improve fraud detection systems.
    You have access to an Isolation Forest model trained on transaction data to detect anomalies.
    When asked about fraud detection, use the fraud alerts and visualization provided to you.
    """,
    llm_config=llm_config,
)

# Create an agent that acts as a customer service representative
customer_service = autogen.AssistantAgent(
    name="Customer_Service_Representative",
    system_message="""You are a customer service representative for a financial organization.
    You can answer common questions about banking services, account management, and fraud reporting.
    Use the knowledge base provided to answer customer questions.
    If you don't know the answer to a question, politely direct the customer to call the customer service line.
    """,
    llm_config=llm_config,
)

# Create an agent that acts as a risk manager
risk_manager = autogen.AssistantAgent(
    name="Risk_Manager",
    system_message="""You are a risk manager for a financial organization.
    You are responsible for assessing and mitigating risks related to fraud, compliance, and operational issues.
    When asked about risk management, provide insights on how to improve the organization's risk posture.
    You can analyze fraud detection results and suggest improvements to the fraud detection system.
    """,
    llm_config=llm_config,
)

# Create a RetrieveAssistantAgent that can access the transaction data and fraud alerts
financial_analyst = RetrieveAssistantAgent(
    name="Financial_Analyst",
    system_message="""You are a financial analyst for a banking organization.
    You have access to transaction data and can analyze it to provide insights on spending patterns, fraud trends, and customer behavior.
    When asked about transaction data, use the provided CSV file to analyze and visualize the data.
    You can also help with fraud detection and risk management by analyzing the transaction data.
    """,
    llm_config=llm_config,
)

# Create a user proxy agent that can interact with the other agents
user_proxy = RetrieveUserProxyAgent(
    name="Bank_Manager",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "qa",
        "docs_path": ".",
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "embedding_model": "text-embedding-ada-002",
        "config_list": config_list,
        "allow_format_str_template": True,
    },
    code_execution_config={"work_dir": "financial_analysis", "use_docker": False},
    llm_config=llm_config,
)

# Define the main function to run the financial organization agents
def run_financial_organization_agents():
    """Run the financial organization agents to analyze fraud detection and customer service."""
    
    # First, have the user proxy initiate a conversation about fraud detection
    user_proxy.initiate_chat(
        fraud_expert,
        message=f"""I need an analysis of the recent fraud detection results. 
        We have detected {len(fraud_alerts)} potentially fraudulent transactions out of {len(df_transactions)} total transactions.
        The accuracy of our fraud detection model is {accuracy:.2%}.
        
        Can you provide insights on these fraud alerts and suggest ways to improve our fraud detection system?
        
        Here are the top 5 fraud alerts:
        {fraud_alerts.head().to_string()}
        
        We also have a visualization of the anomalies in our transaction data at {visualization_path}.
        """
    )
    
    