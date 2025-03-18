
import React from 'react';
import { motion } from 'framer-motion';
import { ChevronDown, AlertTriangle, CheckCircle, AlertCircle } from 'lucide-react';
import { Transaction } from '../types/Transaction';
import { getRiskLevel, formatPercentage } from '../utils/helpers';
import { Button } from './ui/button';

interface TransactionCardProps {
  transaction: Transaction;
  isExpanded: boolean;
  onToggle: () => void;
  onMarkFraud?: (transactionId: number) => void;
  onMarkNonFraud?: (transactionId: number) => void;
}

const TransactionCard: React.FC<TransactionCardProps> = ({ 
  transaction, 
  isExpanded, 
  onToggle,
  onMarkFraud,
  onMarkNonFraud
}) => {
  const riskLevel = getRiskLevel(transaction.Prob_Fraud);
  const [isMarkedFraud, setIsMarkedFraud] = React.useState<boolean | null>(null);
  
  const needsReview = transaction.Prob_Fraud >= 0.7 && transaction.Prob_Fraud <= 0.8;
  
  const RiskIcon = () => {
    switch(riskLevel) {
      case 'high':
        return <AlertTriangle className="h-5 w-5 text-risk-high" />;
      case 'medium':
        return <AlertCircle className="h-5 w-5 text-risk-medium" />;
      case 'low':
        return <CheckCircle className="h-5 w-5 text-risk-low" />;
    }
  };
  
  const riskColors = {
    high: 'from-risk-high/10 to-risk-high/5 border-risk-high/20',
    medium: 'from-risk-medium/10 to-risk-medium/5 border-risk-medium/20',
    low: 'from-risk-low/10 to-risk-low/5 border-risk-low/20'
  };

  const handleMarkFraud = () => {
    setIsMarkedFraud(true);
    if (onMarkFraud) onMarkFraud(transaction.accountNumber);
  };

  const handleMarkNonFraud = () => {
    setIsMarkedFraud(false);
    if (onMarkNonFraud) onMarkNonFraud(transaction.accountNumber);
  };
  
  return (
    <motion.div 
      className={`mb-4 rounded-xl border bg-gradient-to-br ${riskColors[riskLevel]} overflow-hidden hover-scale`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      layout
    >
      <div 
        className="p-4 cursor-pointer"
        onClick={onToggle}
      >
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <RiskIcon />
            <div>
              <h3 className="font-medium">{transaction.merchantName}</h3>
              <p className="text-sm text-gray-500">Customer ID: {transaction.customerId}</p>
            </div>
            
            {riskLevel === 'high' && (
              <div className="ml-2 px-2 py-1 bg-risk-high/10 text-risk-high text-xs font-medium rounded-full border border-risk-high/20">
                Blocked Transaction
              </div>
            )}
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <p className="font-medium">₹{transaction.amount_24h.toFixed(2)}</p>
              <div className={`text-sm font-medium ${
                riskLevel === 'high' ? 'text-risk-high' : 
                riskLevel === 'medium' ? 'text-risk-medium' : 
                'text-risk-low'
              }`}>
                {formatPercentage(transaction.Prob_Fraud)} Risk
              </div>
            </div>
            
            <motion.div
              animate={{ rotate: isExpanded ? 180 : 0 }}
              transition={{ duration: 0.3 }}
            >
              <ChevronDown className="h-5 w-5 text-gray-400" />
            </motion.div>
          </div>
        </div>
      </div>
      
      {isExpanded && (
        <motion.div 
          className="bg-white/50 p-4 border-t border-gray-100"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          transition={{ duration: 0.3 }}
        >
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-500">Account Number</p>
              <p className="font-medium">{transaction.accountNumber}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Account Age</p>
              <p className="font-medium">{transaction.account_age_days} days</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Transactions (24h)</p>
              <p className="font-medium">{transaction.txn_count_24h}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Amount to Avg Ratio</p>
              <p className="font-medium">{transaction.amount_to_avg_ratio.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Distance from Prev</p>
              <p className="font-medium">{transaction.distance_from_prev_txn.toFixed(2)} km</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Foreign Transaction</p>
              <p className="font-medium">{transaction.is_foreign_transaction ? 'Yes' : 'No'}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Different Country</p>
              <p className="font-medium">{transaction.different_country ? 'Yes' : 'No'}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Limit Ratio</p>
              <p className="font-medium">{formatPercentage(transaction.amount_to_limit_ratio)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Available Ratio</p>
              <p className="font-medium">{formatPercentage(transaction.amount_to_available_ratio)}</p>
            </div>
          </div>
          
          {needsReview && (
            <div className="mt-4 flex items-center space-x-3 justify-end">
              <p className="text-sm text-gray-500 mr-2">Mark as:</p>
              <Button 
                variant={isMarkedFraud ? "destructive" : "outline"}
                size="sm"
                onClick={handleMarkFraud}
                className={isMarkedFraud ? "bg-[#ea384c]" : ""}
              >
                Fraud
              </Button>
              <Button 
                variant={isMarkedFraud === false ? "outline" : "outline"}
                size="sm"
                onClick={handleMarkNonFraud}
                className={isMarkedFraud === false ? "bg-[#F2FCE2] border-green-500" : ""}
              >
                Non-Fraud
              </Button>
            </div>
          )}
        </motion.div>
      )}
    </motion.div>
  );
};

export default TransactionCard;
