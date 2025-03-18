
import { Transaction } from '../types/Transaction';

export const mockTransactions: Transaction[] = [
  {
    txn_count_24h: 0,
    amount_to_avg_ratio: 0.5340520931336402,
    amount_24h: 0.0,
    distance_from_prev_txn: 10112.139671074245,
    different_country: 0,
    is_foreign_transaction: 0,
    amount_to_limit_ratio: 0.011456000000000001,
    amount_to_available_ratio: 0.011456000000000001,
    account_age_days: 289,
    days_since_address_change: 289,
    Prob_Fraud: 0.8732585553290808,
    accountNumber: 830329091,
    customerId: 830329091,
    merchantName: "Krispy Kreme #685312"
  },
  {
    txn_count_24h: 2,
    amount_to_avg_ratio: 1.2,
    amount_24h: 420.5,
    distance_from_prev_txn: 5.3,
    different_country: 0,
    is_foreign_transaction: 0,
    amount_to_limit_ratio: 0.05,
    amount_to_available_ratio: 0.08,
    account_age_days: 543,
    days_since_address_change: 120,
    Prob_Fraud: 0.12,
    accountNumber: 239857492,
    customerId: 239857492,
    merchantName: "Amazon Marketplace"
  },
  {
    txn_count_24h: 3,
    amount_to_avg_ratio: 2.7,
    amount_24h: 1240.0,
    distance_from_prev_txn: 3020.75,
    different_country: 1,
    is_foreign_transaction: 1,
    amount_to_limit_ratio: 0.25,
    amount_to_available_ratio: 0.30,
    account_age_days: 89,
    days_since_address_change: 45,
    Prob_Fraud: 0.76,
    accountNumber: 456789123,
    customerId: 456789123,
    merchantName: "AliExpress"
  },
  {
    txn_count_24h: 5,
    amount_to_avg_ratio: 3.9,
    amount_24h: 2580.75,
    distance_from_prev_txn: 0.0,
    different_country: 0,
    is_foreign_transaction: 0,
    amount_to_limit_ratio: 0.45,
    amount_to_available_ratio: 0.55,
    account_age_days: 1095,
    days_since_address_change: 980,
    Prob_Fraud: 0.35,
    accountNumber: 789123456,
    customerId: 789123456,
    merchantName: "Best Buy"
  },
  {
    txn_count_24h: 1,
    amount_to_avg_ratio: 5.6,
    amount_24h: 4500.0,
    distance_from_prev_txn: 8762.3,
    different_country: 1,
    is_foreign_transaction: 1,
    amount_to_limit_ratio: 0.85,
    amount_to_available_ratio: 0.92,
    account_age_days: 45,
    days_since_address_change: 30,
    Prob_Fraud: 0.94,
    accountNumber: 123456789,
    customerId: 123456789,
    merchantName: "Unknown Vendor Ltd"
  }
];

// Generate more transactions for testing pagination
export const generateMockTransactions = (count: number): Transaction[] => {
  const merchants = [
    "Amazon", "Walmart", "Target", "Best Buy", "Apple Store", 
    "Netflix", "Spotify", "Uber", "Lyft", "DoorDash", 
    "Grubhub", "McDonald's", "Starbucks", "7-Eleven", "CVS Pharmacy",
    "Walgreens", "Home Depot", "Lowe's", "IKEA", "Costco"
  ];
  
  const result: Transaction[] = [];
  
  for (let i = 0; i < count; i++) {
    const probFraud = Math.random();
    const isForeign = Math.random() > 0.8 ? 1 : 0;
    const isDifferentCountry = isForeign === 1 && Math.random() > 0.5 ? 1 : 0;
    
    result.push({
      txn_count_24h: Math.floor(Math.random() * 10),
      amount_to_avg_ratio: Math.random() * 5 + 0.1,
      amount_24h: Math.random() * 5000,
      distance_from_prev_txn: Math.random() * 15000,
      different_country: isDifferentCountry,
      is_foreign_transaction: isForeign,
      amount_to_limit_ratio: Math.random() * 0.9,
      amount_to_available_ratio: Math.random() * 0.9,
      account_age_days: Math.floor(Math.random() * 1500),
      days_since_address_change: Math.floor(Math.random() * 500),
      Prob_Fraud: probFraud,
      accountNumber: 100000000 + Math.floor(Math.random() * 900000000),
      customerId: 100000000 + Math.floor(Math.random() * 900000000),
      merchantName: `${merchants[Math.floor(Math.random() * merchants.length)]} #${Math.floor(Math.random() * 1000000)}`
    });
  }
  
  return result;
};

export const extendedMockTransactions = [...mockTransactions, ...generateMockTransactions(25)];
