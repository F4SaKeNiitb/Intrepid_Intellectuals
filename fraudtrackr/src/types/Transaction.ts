
export interface Transaction {
  txn_count_24h: number;
  amount_to_avg_ratio: number;
  amount_24h: number;
  distance_from_prev_txn: number;
  different_country: 0 | 1;
  is_foreign_transaction: 0 | 1;
  amount_to_limit_ratio: number;
  amount_to_available_ratio: number;
  account_age_days: number;
  days_since_address_change: number;
  Prob_Fraud: number;
  accountNumber: number;
  customerId: number;
  merchantName: string;
}

export interface TransactionFilters {
  searchTerm: string;
  minFraudProb: number;
  maxFraudProb: number;
  isForeignOnly: boolean;
  isDifferentCountryOnly: boolean;
  merchantName: string;
  riskCategories?: ('red' | 'yellow' | 'green')[];
}
