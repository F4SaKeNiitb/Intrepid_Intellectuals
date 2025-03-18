
import React, { useMemo } from 'react';
import { Transaction } from '../types/Transaction';
import { formatPercentage } from '../utils/helpers';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import { BarChart3, PieChart as PieChartIcon } from 'lucide-react';

interface DashboardProps {
  transactions: Transaction[];
}

const Dashboard: React.FC<DashboardProps> = ({ transactions }) => {
  // Risk distribution data for pie chart
  const riskDistribution = useMemo(() => {
    const highRisk = transactions.filter(t => t.Prob_Fraud > 0.8).length;
    const mediumRisk = transactions.filter(t => t.Prob_Fraud >= 0.5 && t.Prob_Fraud <= 0.8).length;
    const lowRisk = transactions.filter(t => t.Prob_Fraud < 0.5).length;
    
    return [
      { name: 'High Risk', value: highRisk, color: '#DA251C' },
      { name: 'Medium Risk', value: mediumRisk, color: '#FFA500' },
      { name: 'Low Risk', value: lowRisk, color: '#10B981' }
    ];
  }, [transactions]);

  // Transaction trends (foreign vs domestic)
  const transactionTypes = useMemo(() => {
    const foreign = transactions.filter(t => t.is_foreign_transaction === 1).length;
    const domestic = transactions.filter(t => t.is_foreign_transaction === 0).length;
    
    return [
      { name: 'Foreign', value: foreign, color: '#00579C' },
      { name: 'Domestic', value: domestic, color: '#ccc' }
    ];
  }, [transactions]);

  // Total transactions and high risk stats
  const stats = useMemo(() => {
    const total = transactions.length;
    const highRisk = transactions.filter(t => t.Prob_Fraud > 0.8).length;
    const averageFraudProb = transactions.length ? 
      transactions.reduce((sum, t) => sum + t.Prob_Fraud, 0) / transactions.length : 
      0;
    
    return {
      total,
      highRisk,
      highRiskPercentage: total ? (highRisk / total) * 100 : 0,
      averageFraudProb
    };
  }, [transactions]);

  return (
    <motion.div 
      className="glass rounded-xl p-4 h-full"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-xl font-semibold mb-4">Dashboard</h2>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="glass-dark p-4 rounded-lg">
          <p className="text-sm text-gray-500">Total Transactions</p>
          <p className="text-2xl font-semibold">{stats.total}</p>
        </div>
        
        <div className="glass-dark p-4 rounded-lg">
          <p className="text-sm text-gray-500">High Risk</p>
          <p className="text-2xl font-semibold text-risk-high">{stats.highRisk}</p>
        </div>
        
        <div className="glass-dark p-4 rounded-lg">
          <p className="text-sm text-gray-500">High Risk %</p>
          <p className="text-2xl font-semibold">{stats.highRiskPercentage.toFixed(1)}%</p>
        </div>
        
        <div className="glass-dark p-4 rounded-lg">
          <p className="text-sm text-gray-500">Avg Fraud Prob</p>
          <p className="text-2xl font-semibold">{formatPercentage(stats.averageFraudProb)}</p>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white/50 p-4 rounded-lg">
          <h3 className="text-sm font-medium flex items-center mb-2">
            <PieChartIcon className="h-4 w-4 mr-1 text-unionbank-blue" />
            Risk Distribution
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                >
                  {riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="bg-white/50 p-4 rounded-lg">
          <h3 className="text-sm font-medium flex items-center mb-2">
            <BarChart3 className="h-4 w-4 mr-1 text-unionbank-blue" />
            Transaction Types
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={transactionTypes}>
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {transactionTypes.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default Dashboard;
