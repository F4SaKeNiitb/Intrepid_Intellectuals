
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Header from '../components/Header';
import TransactionList from '../components/TransactionList';
import Dashboard from '../components/Dashboard';
import Filters from '../components/Filters';
import FileLoader from '../components/FileLoader';
import ExportButton from '../components/ExportButton';
import { Transaction, TransactionFilters } from '../types/Transaction';
import { filterTransactions } from '../utils/helpers';
import { mockTransactions } from '../utils/mockData';

const defaultFilters: TransactionFilters = {
  searchTerm: '',
  minFraudProb: 0,
  maxFraudProb: 1,
  isForeignOnly: false,
  isDifferentCountryOnly: false,
  merchantName: ''
};

const Index = () => {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [filteredTransactions, setFilteredTransactions] = useState<Transaction[]>([]);
  const [filters, setFilters] = useState<TransactionFilters>(defaultFilters);
  const [isLoading, setIsLoading] = useState(true);

  // Initial load effect
  useEffect(() => {
    // Simulate initial data loading
    const timer = setTimeout(() => {
      setTransactions(mockTransactions);
      setIsLoading(false);
    }, 1500);

    return () => clearTimeout(timer);
  }, []);

  // Filter transactions when filters or transactions change
  useEffect(() => {
    setFilteredTransactions(filterTransactions(transactions, filters));
  }, [transactions, filters]);

  // Handle new transaction data
  const handleDataLoaded = (transaction: Transaction) => {
    setTransactions(prev => [transaction, ...prev]);
  };

  // Handle filter changes
  const handleFilterChange = (newFilters: TransactionFilters) => {
    setFilters(newFilters);
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] p-4 md:p-6">
      <div className="max-w-[1400px] mx-auto">
        <Header />

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2 space-y-6">
            <Filters onFilterChange={handleFilterChange} />
            <FileLoader onDataLoaded={handleDataLoaded} />
            <TransactionList 
              transactions={filteredTransactions} 
              isLoading={isLoading}
            />
          </div>

          <div className="space-y-6">
            <Dashboard transactions={filteredTransactions} />
            <ExportButton transactions={filteredTransactions} />

            <motion.div 
              className="glass rounded-xl p-4 text-center text-sm text-gray-500"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1, duration: 0.5 }}
            >
              <div className="flex justify-center mb-2">
                <img 
                  src="/lovable-uploads/2a9c9e91-fcab-441e-ba02-1e4c6dca548a.png" 
                  alt="Union Bank Banner" 
                  className="h-8" 
                />
              </div>
              <p>Union Bank of India Fraud Detection System</p>
              <p>© {new Date().getFullYear()} Union Bank of India</p>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
