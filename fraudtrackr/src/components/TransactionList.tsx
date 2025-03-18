
import React, { useState, useEffect } from 'react';
import { LayoutGrid, AlertTriangle } from 'lucide-react';
import { Transaction } from '../types/Transaction';
import { sortTransactionsByRisk } from '../utils/helpers';
import TransactionCard from './TransactionCard';
import { motion } from 'framer-motion';

interface TransactionListProps {
  transactions: Transaction[];
  isLoading: boolean;
}

const TransactionList: React.FC<TransactionListProps> = ({ 
  transactions, 
  isLoading 
}) => {
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [sortedTransactions, setSortedTransactions] = useState<Transaction[]>([]);
  const [page, setPage] = useState(1);
  const [markedFraud, setMarkedFraud] = useState<{[key: number]: boolean}>({});
  const itemsPerPage = 10;

  useEffect(() => {
    // Sort transactions by risk (highest first)
    setSortedTransactions(sortTransactionsByRisk(transactions));
  }, [transactions]);

  const toggleExpand = (id: number) => {
    setExpandedId(expandedId === id ? null : id);
  };

  const handleMarkFraud = (transactionId: number) => {
    setMarkedFraud(prev => ({...prev, [transactionId]: true}));
  };

  const handleMarkNonFraud = (transactionId: number) => {
    setMarkedFraud(prev => ({...prev, [transactionId]: false}));
  };

  const totalPages = Math.ceil(sortedTransactions.length / itemsPerPage);
  const displayedTransactions = sortedTransactions.slice(
    0, 
    page * itemsPerPage
  );

  return (
    <div className="glass rounded-xl p-4 h-full">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold flex items-center">
          <LayoutGrid className="h-5 w-5 mr-2 text-unionbank-blue" />
          Transactions
          {sortedTransactions.length > 0 && (
            <span className="ml-2 text-sm text-gray-500">
              ({sortedTransactions.length})
            </span>
          )}
        </h2>
      </div>

      <div className="overflow-y-auto scrollbar-thin" style={{ maxHeight: 'calc(100vh - 280px)' }}>
        {isLoading && transactions.length === 0 ? (
          <div className="h-40 flex flex-col items-center justify-center text-gray-400">
            <div className="w-8 h-8 border-4 border-unionbank-blue/30 border-t-unionbank-blue rounded-full animate-spin mb-3"></div>
            <p>Loading transactions...</p>
          </div>
        ) : transactions.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-gray-400">
            <AlertTriangle className="h-10 w-10 mb-2" />
            <p>No transactions found</p>
          </div>
        ) : (
          <>
            {displayedTransactions.map((transaction, index) => (
              <TransactionCard
                key={`${transaction.accountNumber}-${index}`}
                transaction={transaction}
                isExpanded={expandedId === index}
                onToggle={() => toggleExpand(index)}
                onMarkFraud={handleMarkFraud}
                onMarkNonFraud={handleMarkNonFraud}
              />
            ))}
            
            {page < totalPages && (
              <motion.button
                className="w-full py-3 text-center text-unionbank-blue hover:bg-gray-50 rounded-md transition"
                onClick={() => setPage(page + 1)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Load More
              </motion.button>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default TransactionList;
