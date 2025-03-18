
import React from 'react';
import { Download, AlertTriangle } from 'lucide-react';
import { Transaction } from '../types/Transaction';
import { exportToCSV, downloadCSV } from '../utils/helpers';
import { motion } from 'framer-motion';
import { toast } from "sonner";

interface ExportButtonProps {
  transactions: Transaction[];
}

const ExportButton: React.FC<ExportButtonProps> = ({ transactions }) => {
  const highRiskTransactions = transactions.filter(t => t.Prob_Fraud > 0.8);

  const handleExport = () => {
    if (highRiskTransactions.length === 0) {
      toast.error("No high-risk transactions to export", {
        description: "No transactions with fraud probability > 0.8 found."
      });
      return;
    }

    const csv = exportToCSV(highRiskTransactions);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    downloadCSV(csv, `high-risk-transactions-${timestamp}.csv`);
    
    toast.success("Export successful", {
      description: `${highRiskTransactions.length} high-risk transactions exported to CSV.`
    });
  };

  return (
    <motion.button
      className="glass rounded-xl p-4 flex items-center justify-between w-full"
      onClick={handleExport}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="flex items-center">
        <div className="bg-unionbank-red/10 p-2 rounded-full mr-3">
          <AlertTriangle className="h-5 w-5 text-unionbank-red" />
        </div>
        <div className="text-left">
          <h3 className="font-medium">Export High-Risk Transactions</h3>
          <p className="text-sm text-gray-500">
            {highRiskTransactions.length} transactions with fraud probability &gt; 80%
          </p>
        </div>
      </div>
      <Download className="h-5 w-5 text-unionbank-blue" />
    </motion.button>
  );
};

export default ExportButton;
