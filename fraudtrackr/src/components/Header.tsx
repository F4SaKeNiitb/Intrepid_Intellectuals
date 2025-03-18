
import React from 'react';
import { motion } from 'framer-motion';

const Header: React.FC = () => {
  return (
    <motion.header 
      className="glass flex items-center justify-between p-4 mb-6 rounded-xl"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="flex items-center">
        <img 
          src="/lovable-uploads/909a7f25-0763-4f58-ac0b-18bbeafd3e5c.png" 
          alt="Union Bank Logo" 
          className="h-12 w-auto mr-4" 
        />
        <div>
          <h1 className="text-unionbank-blue font-semibold text-xl">Fraud Detection System</h1>
          <p className="text-gray-500 text-sm">Credit Card Transaction Monitor</p>
        </div>
      </div>
      
      <div className="flex items-center space-x-4">
        <div className="flex items-center glass-dark py-1 px-3 rounded-full text-sm">
          <div className="h-2 w-2 rounded-full bg-green-500 mr-2 animate-pulse"></div>
          <span>Live Monitoring</span>
        </div>
      </div>
    </motion.header>
  );
};

export default Header;
