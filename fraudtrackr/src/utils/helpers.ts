
import { Transaction, TransactionFilters } from "../types/Transaction";

// Format currency values
export const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0
  }).format(amount);
};

// Get risk level based on fraud probability
export const getRiskLevel = (probFraud: number): 'high' | 'medium' | 'low' => {
  if (probFraud > 0.8) return 'high';
  if (probFraud >= 0.5) return 'medium';
  return 'low';
};

// Get risk category for filtering
export const getRiskCategory = (probFraud: number): 'red' | 'yellow' | 'green' => {
  if (probFraud > 0.8) return 'red';
  if (probFraud >= 0.5) return 'yellow';
  return 'green';
};

// Get risk color based on fraud probability
export const getRiskColor = (probFraud: number): string => {
  if (probFraud > 0.8) return 'risk-high';
  if (probFraud >= 0.5) return 'risk-medium';
  return 'risk-low';
};

// Format percentage
export const formatPercentage = (value: number): string => {
  return `${(value * 100).toFixed(2)}%`;
};

// Filter transactions based on filters
export const filterTransactions = (
  transactions: Transaction[],
  filters: TransactionFilters
): Transaction[] => {
  return transactions.filter((transaction) => {
    // Search term filter
    if (
      filters.searchTerm &&
      !Object.values(transaction).some((value) =>
        String(value).toLowerCase().includes(filters.searchTerm.toLowerCase())
      )
    ) {
      return false;
    }

    // Fraud probability range
    if (
      transaction.Prob_Fraud < filters.minFraudProb ||
      transaction.Prob_Fraud > filters.maxFraudProb
    ) {
      return false;
    }

    // Foreign transaction filter
    if (filters.isForeignOnly && transaction.is_foreign_transaction !== 1) {
      return false;
    }

    // Different country filter
    if (filters.isDifferentCountryOnly && transaction.different_country !== 1) {
      return false;
    }

    // Merchant name filter
    if (
      filters.merchantName &&
      !transaction.merchantName
        .toLowerCase()
        .includes(filters.merchantName.toLowerCase())
    ) {
      return false;
    }

    // Risk category filter
    if (filters.riskCategories && filters.riskCategories.length > 0) {
      const category = getRiskCategory(transaction.Prob_Fraud);
      if (!filters.riskCategories.includes(category)) {
        return false;
      }
    }

    return true;
  });
};

// Sort transactions by fraud probability (highest first)
export const sortTransactionsByRisk = (transactions: Transaction[]): Transaction[] => {
  return [...transactions].sort((a, b) => b.Prob_Fraud - a.Prob_Fraud);
};

// Export transactions to CSV
export const exportToCSV = (transactions: Transaction[]): string => {
  const headers = Object.keys(transactions[0]).join(',');
  const rows = transactions.map(transaction => 
    Object.values(transaction).map(value => 
      typeof value === 'string' ? `"${value}"` : value
    ).join(',')
  ).join('\n');
  
  return `${headers}\n${rows}`;
};

// Download CSV file
export const downloadCSV = (csvContent: string, filename: string): void => {
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

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

// Simulate loading JSON files
export const simulateJSONFileLoading = (
  onDataLoaded: (data: Transaction) => void, 
  interval: number = 500
): { start: () => void; stop: () => void } => {
  let intervalId: number | null = null;
  
  const start = () => {
    const allData = [...generateMockTransactions(50)];
    let index = 0;
    
    intervalId = window.setInterval(() => {
      if (index < allData.length) {
        onDataLoaded(allData[index]);
        index++;
      } else {
        if (intervalId !== null) {
          clearInterval(intervalId);
        }
      }
    }, interval);
  };
  
  const stop = () => {
    if (intervalId !== null) {
      clearInterval(intervalId);
      intervalId = null;
    }
  };
  
  return { start, stop };
};
